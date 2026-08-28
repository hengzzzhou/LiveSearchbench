#!/usr/bin/env python3
"""Level 3 (abstracted) question generation.

Level 3 rewrites a Level 2 question so that the entities it names are replaced
by indirect descriptions ("node expansion"), which makes the question harder to
answer by surface matching while leaving the answer unchanged.

Because the answer is unchanged, the verification program is inherited from the
Level 2 parent. The previous release inherited it silently *and* wrote
``"answer_verified": True`` on every instance while the uniqueness check itself
was commented out. Here:

* the check runs again by default (``--verify count``), against the endpoint
  named in the metadata;
* ``answer_verified`` reports what actually happened and is never a literal
  ``True``;
* every instance carries ``sparql_verification_source: "inherited_from_level2"``
  and a ``verification`` block recording the mode, the endpoint, the timestamp
  and the observed result count;
* ``--verify skip`` is still available, but then writes ``answer_verified:
  false`` rather than claiming a check that did not run.

Descriptions are drawn from the input CSV and from Wikidata. Predicates are
screened against :data:`livesearchbench.filters.EXCLUDED_PROPERTY_IDS` so that
meta/formatting properties never leak into a description; the narrower 198-label
relation allow-list is deliberately *not* applied here, because it governs which
relations may carry a question, not which facts may describe an entity.

Examples:
    python scripts/generate_level3.py --input data/sample/triple_changes_sample.csv \
        --level2 outputs/questions/smoketest_level2.json \
        --dry-run --num-questions 5 --output outputs/questions/demo_level3.json

    python scripts/generate_level3.py --input outputs/extracted_triples/triple_changes.csv \
        --level2 outputs/questions/level2_300_questions_2025.json \
        --model gpt-4o --num-questions 200 --seed 0 --verify answer
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from livesearchbench import config, filters, scoring
from livesearchbench.config import MissingCredential
from livesearchbench.dataio import DatasetFormatError, ensure_parent, load_instances
from livesearchbench.http import PoliteSession, RequestFailed
from livesearchbench.sparql import SparqlClient, SparqlError, to_label_select

logger = logging.getLogger("generate_level3")

#: Shared default across the three generators; override with --model.
DEFAULT_MODEL = "gpt-4o"
DEFAULT_NUM_QUESTIONS = 200

EXTRACTOR_COLUMNS = ("entity_id", "entity_label", "property_id", "property_label", "new_value")
TRIPLE_COLUMNS = ("subject_id", "subject_label", "predicate_id", "predicate_label",
                  "object_id", "object_label")

FALLBACK_INPUTS = (
    PROJECT_ROOT / "outputs" / "extracted_triples" / "triple_changes_latest.csv",
    PROJECT_ROOT / "data" / "final_changed_item_with_id.csv",
    PROJECT_ROOT / "data" / "sample" / "triple_changes_sample.csv",
)

#: Descriptive properties queried when expanding an entity into indirect
#: descriptions. Deny-listed IDs are removed, so P31/P279 cannot appear.
DESCRIPTIVE_PROPERTY_IDS: Tuple[str, ...] = tuple(
    pid for pid in (
        "P27", "P19", "P20", "P103", "P136", "P106", "P495", "P37", "P36", "P571",
        "P577", "P57", "P50", "P175", "P1412", "P17", "P131", "P276", "P159",
        "P140", "P108", "P69", "P54", "P166", "P127", "P449", "P123", "P364",
        "P30", "P38", "P122", "P735", "P734", "P4552",
    ) if pid not in filters.EXCLUDED_PROPERTY_IDS
)

#: Properties used by the reverse-hop abstraction (C --p--> V, describe V by C).
REVERSE_HOP_PROPERTY_IDS: Tuple[str, ...] = tuple(
    pid for pid in (
        "P30", "P17", "P131", "P361", "P276", "P150", "P706", "P527", "P190",
        "P47", "P206", "P138", "P170", "P178", "P272", "P264", "P449", "P750",
        "P123", "P127", "P1830", "P355", "P749", "P112", "P37", "P103", "P1412",
        "P364", "P407", "P495", "P840", "P915", "P291", "P159", "P740",
    ) if pid not in filters.EXCLUDED_PROPERTY_IDS
)

#: Phrasings for the reverse-hop descriptions, keyed by property label.
REVERSE_HOP_PHRASINGS: Dict[str, str] = {
    "continent": "{label} entity",
    "country": "{label} based",
    "contains the administrative territorial entity": "{label} region",
    "located on terrain feature": "{label} region",
    "shares border with": "neighbour of {label}",
    "twinned administrative body": "neighbour of {label}",
    "located in or next to body of water": "{label} adjacent",
    "parent organization": "{label} affiliate",
    "owned by": "{label} affiliate",
    "founded by": "{label} affiliate",
    "subsidiary": "{label} branch",
    "has part": "{label} branch",
    "production company": "{label} production",
    "record label": "{label} production",
    "publisher": "{label} production",
    "distributed by": "{label} production",
    "original broadcaster": "{label} network",
    "creator": "{label} creation",
    "developer": "{label} creation",
    "named after": "namesake of {label}",
    "official language": "{label} speaking",
    "native language": "{label} speaking",
    "original language of film or TV show": "{label} speaking",
    "filming location": "{label} associated",
    "narrative location": "{label} associated",
    "place of publication": "{label} associated",
    "headquarters location": "{label} established",
    "location of formation": "{label} established",
    "part of": "{label}",
}

ABSTRACT_CONCEPTS = frozenset({"profession", "occupation", "concept", "category",
                               "classification", "type", "kind", "form", "class"})


class QuestionWriter:
    """Chat-completions client with a deterministic offline stub."""

    def __init__(self, *, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 dry_run: bool = False, temperature: float = 0.3, max_tokens: int = 256,
                 max_attempts: int = 3) -> None:
        self.model = model
        self.dry_run = dry_run
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0
        self.failures = 0
        self._session: Optional[PoliteSession] = None
        self._endpoint = ""
        self._api_key = ""
        if not dry_run:
            base, key = config.openai_credentials(base_url=base_url, api_key=api_key)
            base = base.rstrip("/")
            self._endpoint = base if base.endswith("/chat/completions") else base + "/chat/completions"
            self._api_key = key
            self._session = PoliteSession(component="LiveSearchBench-L3", max_attempts=max_attempts)

    @property
    def source_label(self) -> str:
        return "template_stub" if self.dry_run else "llm"

    @property
    def model_label(self) -> str:
        return "template-stub (dry run)" if self.dry_run else self.model

    def write(self, messages: Sequence[Dict[str, str]], *, stub: str) -> Optional[str]:
        self.calls += 1
        if self.dry_run:
            return stub
        payload = {"model": self.model, "messages": list(messages),
                   "temperature": self.temperature, "max_tokens": self.max_tokens}
        try:
            response = self._session.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}",
                         "Content-Type": "application/json"},
                json=payload,
            )
        except RequestFailed as exc:
            logger.warning("chat completion failed: %s", exc)
            self.failures += 1
            return None
        if response.status_code != 200:
            logger.warning("chat completion HTTP %d: %s", response.status_code, response.text[:200])
            self.failures += 1
            return None
        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            logger.warning("unexpected chat completion payload: %s", exc)
            self.failures += 1
            return None

    def close(self) -> None:
        if self._session is not None:
            self._session.close()


def resolve_input(argument: Optional[str]) -> Path:
    """Return the triple CSV to read, honouring the historical search order."""
    if argument:
        path = Path(argument)
        if not path.is_file():
            raise SystemExit(f"Input file not found: {path}")
        return path
    for candidate in FALLBACK_INPUTS:
        if candidate.is_file():
            logger.info("No --input given; using %s", candidate)
            return candidate
    raise SystemExit(
        "No input CSV given and none of the default locations exist:\n  "
        + "\n  ".join(str(c) for c in FALLBACK_INPUTS)
    )


def resolve_level2(argument: Optional[str]) -> Path:
    """Return the Level 2 file, defaulting to the newest one in outputs/questions."""
    if argument:
        path = Path(argument)
        if not path.is_file():
            raise SystemExit(
                f"Level 2 file not found: {path}\n"
                f"  Generate one with scripts/generate_level2.py --output {path}"
            )
        return path
    directory = PROJECT_ROOT / "outputs" / "questions"
    matches = sorted(directory.glob("level2_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    matches = [m for m in matches if not m.name.endswith(".run_report.json")]
    if not matches:
        raise SystemExit(
            "No --level2 file given and none found in outputs/questions/.\n"
            "  Run scripts/generate_level2.py first, then pass its output with --level2."
        )
    logger.info("No --level2 given; using the newest match %s", matches[0])
    return matches[0]


def load_triples(path: Path) -> pd.DataFrame:
    """Load the triple CSV used as a local source of entity descriptions."""
    frame = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(frame), path)
    if "new_value" in frame.columns:
        missing = [c for c in EXTRACTOR_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} looks like extractor output but lacks columns: {missing}")
        entity_valued = frame[frame["new_value"].astype(str).str.match(r"^Q\d+$", na=False)]
        frame = pd.DataFrame({
            "subject_id": entity_valued["entity_id"],
            "subject_label": entity_valued["entity_label"],
            "predicate_id": entity_valued["property_id"],
            "predicate_label": entity_valued["property_label"],
            "object_id": entity_valued["new_value"],
            "object_label": entity_valued["new_value_label"],
        })
    else:
        missing = [c for c in TRIPLE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(
                f"{path} is neither extractor output (needs 'new_value') nor a triple CSV "
                f"(missing {missing})."
            )
        frame = frame[list(TRIPLE_COLUMNS)]
    frame = frame.dropna(subset=["subject_id", "predicate_id", "object_id"]).astype(str)
    if frame.empty:
        raise ValueError(f"{path} yielded zero usable triples after format conversion.")
    return frame


QID_PATTERN = re.compile(r"^Q\d+$")


def resolve_surface_forms(client: SparqlClient, values: Sequence[str]) -> List[str]:
    """Expand bare QIDs into their English label and aliases.

    ``SparqlClient.select_labels`` falls back to the bare QID when the label
    service returns nothing, which it does for ``SELECT DISTINCT`` projections.
    Comparing a QID against a gold string would fail every time, so the labels
    are fetched explicitly here, aliases included, before the comparison.
    """
    out: List[str] = []
    for value in values:
        if not QID_PATTERN.match(value):
            out.append(value)
            continue
        query = (f'SELECT ?form WHERE {{\n'
                 f'  {{ wd:{value} rdfs:label ?form . }} UNION '
                 f'{{ wd:{value} skos:altLabel ?form . }}\n'
                 f'  FILTER(LANG(?form) = "en")\n}}\nLIMIT 20')
        try:
            data = client.query(query)
        except SparqlError as exc:
            logger.warning("label lookup for %s failed: %s", value, exc)
            out.append(value)
            continue
        forms = [row["form"]["value"] for row in data.get("results", {}).get("bindings", [])
                 if row.get("form", {}).get("value")]
        out.extend(forms or [value])
    return out


class NodeExpansionEngine:
    """Turn an entity name into indirect descriptions of that entity."""

    def __init__(self, frame: pd.DataFrame, client: Optional[SparqlClient], *,
                 source: str, max_properties: int, stats: filters.FilterStats) -> None:
        self.frame = frame
        self.client = client
        self.source = source
        self.max_properties = max_properties
        self.stats = stats
        self._cache: Dict[str, List[str]] = {}

    def find_entity_id(self, entity_label: str) -> Optional[str]:
        """Look the label up in the triple table (either end of a triple)."""
        matches = self.frame[(self.frame["subject_label"] == entity_label)
                             | (self.frame["object_label"] == entity_label)]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return row["subject_id"] if row["subject_label"] == entity_label else row["object_id"]

    def describe(self, entity_label: str) -> List[str]:
        """Return indirect descriptions for ``entity_label``, best first."""
        if entity_label in self._cache:
            return self._cache[entity_label]
        descriptions: List[str] = []
        entity_id = self.find_entity_id(entity_label)

        if self.source in ("auto", "wikidata") and entity_id and self.client is not None:
            descriptions.extend(self.wikidata_descriptions(entity_id, entity_label))
        if self.source in ("auto", "local") and not descriptions:
            descriptions.extend(self.local_descriptions(entity_label))
        if self.source == "reverse_hop" or (self.source == "auto" and not descriptions):
            if entity_id and self.client is not None:
                descriptions.extend(self.reverse_hop_descriptions(entity_id, entity_label))

        # Deduplicate while keeping order.
        seen, unique = set(), []
        for description in descriptions:
            text = description.strip()
            if text and text.lower() not in seen:
                seen.add(text.lower())
                unique.append(text)
        self._cache[entity_label] = unique
        return unique

    def local_descriptions(self, entity_label: str) -> List[str]:
        """Descriptions taken from the input CSV only (no network)."""
        out: List[str] = []
        as_subject = self.frame[self.frame["subject_label"] == entity_label].head(5)
        for row in as_subject.itertuples(index=False):
            if not filters.is_allowed_relation(str(row.predicate_id), use_allowlist=False):
                continue
            out.append(f"has {row.predicate_label} {row.object_label}")
        as_object = self.frame[self.frame["object_label"] == entity_label].head(5)
        for row in as_object.itertuples(index=False):
            if not filters.is_allowed_relation(str(row.predicate_id), use_allowlist=False):
                continue
            out.append(f"is the {row.predicate_label} of {row.subject_label}")
        return out

    def wikidata_descriptions(self, entity_id: str, entity_label: str) -> List[str]:
        """Descriptions built from descriptive Wikidata statements."""
        values = " ".join(f"wd:{pid}" for pid in DESCRIPTIVE_PROPERTY_IDS)
        query = f"""
SELECT ?prop ?propLabel ?value ?valueLabel WHERE {{
  VALUES ?prop {{ {values} }}
  ?prop wikibase:directClaim ?property .
  wd:{entity_id} ?property ?value .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {max(self.max_properties * 4, 20)}
"""
        try:
            data = self.client.query(query)
        except SparqlError as exc:
            self.stats.drop("expansion_sparql_error")
            logger.warning("description query failed for %s: %s", entity_id, exc)
            return []

        described: List[str] = []
        for row in data.get("results", {}).get("bindings", []):
            property_id = row.get("prop", {}).get("value", "").rsplit("/", 1)[-1]
            property_label = row.get("propLabel", {}).get("value", "") or property_id
            value_label = simplify_value(row.get("valueLabel", {}).get("value", ""))
            if not filters.is_allowed_relation(property_id, use_allowlist=False):
                continue
            if not is_usable_value(property_label, value_label):
                continue
            if value_label.strip().lower() == entity_label.strip().lower():
                continue
            described.append(f"({entity_label},{property_label},{value_label})")
        return select_diverse(described, self.max_properties)

    def reverse_hop_descriptions(self, entity_id: str, entity_label: str) -> List[str]:
        """Describe the entity through entities that point at it.

        The subject of the reverse hop is bound by QID rather than by an
        ``rdfs:label`` lookup; matching on the label made this query scan a
        large part of the graph and time out.
        """
        values = " ".join(f"wd:{pid}" for pid in REVERSE_HOP_PROPERTY_IDS)
        query = f"""
SELECT ?reverseEntity ?reverseEntityLabel ?prop ?propLabel WHERE {{
  VALUES ?prop {{ {values} }}
  ?prop wikibase:directClaim ?property .
  ?reverseEntity ?property wd:{entity_id} .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 10
"""
        try:
            data = self.client.query(query)
        except SparqlError as exc:
            self.stats.drop("expansion_sparql_error")
            logger.warning("reverse-hop query failed for %s: %s", entity_label, exc)
            return []

        out: List[str] = []
        for row in data.get("results", {}).get("bindings", []):
            reverse_label = row.get("reverseEntityLabel", {}).get("value", "")
            property_label = row.get("propLabel", {}).get("value", "")
            if not (2 < len(reverse_label) < 50) or QID_PATTERN.match(reverse_label):
                continue
            template = REVERSE_HOP_PHRASINGS.get(property_label)
            if template:
                out.append(template.format(label=reverse_label))
        return out


def simplify_value(value_label: str) -> str:
    """Render a Wikidata datetime literal as a plain year."""
    match = re.match(r"^[+-]?0*(\d{1,4})-\d{2}-\d{2}T", value_label or "")
    return match.group(1) if match else value_label


def is_usable_value(property_label: str, value_label: str) -> bool:
    """Reject URIs, bare QIDs, over-long strings and empty abstractions."""
    if not property_label or not value_label:
        return False
    if value_label.startswith("Q") and value_label[1:].isdigit():
        return False
    if "http://" in property_label or "https://" in property_label:
        return False
    if "http://" in value_label.lower() or "https://" in value_label.lower():
        return False
    if len(value_label) > 100 or value_label.count("/") > 1:
        return False
    return value_label.lower() not in ABSTRACT_CONCEPTS


def select_diverse(descriptions: Sequence[str], limit: int) -> List[str]:
    """Keep at most one description per property, shortest value first."""
    by_property: Dict[str, str] = {}
    for description in descriptions:
        parts = description.strip("()").split(",")
        key = parts[1] if len(parts) > 2 else description
        current = by_property.get(key)
        if current is None or len(description) < len(current):
            by_property[key] = description
    return list(by_property.values())[:limit]


class Level3Generator:
    """Rewrite Level 2 questions into abstracted Level 3 questions."""

    def __init__(self, *, engine: NodeExpansionEngine, writer: QuestionWriter,
                 client: Optional[SparqlClient], verify: str, keep_unverified: bool,
                 max_constraints: int, rng: random.Random,
                 stats: filters.FilterStats, endpoint: Optional[str]) -> None:
        self.engine = engine
        self.writer = writer
        self.client = client
        self.verify = verify
        self.keep_unverified = keep_unverified
        self.max_constraints = max_constraints
        self.rng = rng
        self.stats = stats
        self.endpoint = endpoint

    # -- verification ------------------------------------------------------

    def verify_answer(self, sparql: str, expected_answer: str,
                      parent_flag: Optional[bool]) -> Dict:
        """Run the inherited program and report exactly what was observed."""
        checked = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if self.verify == "skip":
            return {"mode": "skip", "verified": False, "checked_utc": None,
                    "endpoint": None, "result_count": None,
                    "note": "no check was run (--verify skip)"}
        if self.verify == "inherit":
            return {"mode": "inherit", "verified": bool(parent_flag), "checked_utc": None,
                    "endpoint": None, "result_count": None,
                    "note": "copied from the Level 2 parent's own verification flag"}
        if not sparql:
            return {"mode": self.verify, "verified": False, "checked_utc": checked,
                    "endpoint": self.endpoint, "result_count": None,
                    "note": "the Level 2 parent carried no SPARQL program"}
        try:
            count = self.client.count(sparql)
        except SparqlError as exc:
            self.stats.drop("sparql_error")
            return {"mode": self.verify, "verified": False, "checked_utc": checked,
                    "endpoint": self.endpoint, "result_count": None,
                    "note": f"verification request failed: {exc}"}
        if count != 1:
            return {"mode": self.verify, "verified": False, "checked_utc": checked,
                    "endpoint": self.endpoint, "result_count": count,
                    "note": "the inherited program no longer has a unique answer"}
        if self.verify == "count":
            return {"mode": "count", "verified": True, "checked_utc": checked,
                    "endpoint": self.endpoint, "result_count": 1,
                    "note": "COUNT == 1 on the endpoint above"}

        select = to_label_select(sparql)
        if not select:
            return {"mode": "answer", "verified": False, "checked_utc": checked,
                    "endpoint": self.endpoint, "result_count": 1,
                    "note": "the inherited program is not a COUNT query, so the returned "
                            "entity could not be compared with the gold answer"}
        try:
            labels = self.client.select_labels(select)
        except SparqlError as exc:
            self.stats.drop("sparql_error")
            return {"mode": "answer", "verified": False, "checked_utc": checked,
                    "endpoint": self.endpoint, "result_count": 1,
                    "note": f"label lookup failed: {exc}"}
        surface_forms = resolve_surface_forms(self.client, labels)
        matched = any(scoring.exact_match(form, expected_answer) for form in surface_forms)
        if matched:
            note = "the endpoint returns the gold answer"
        elif all(QID_PATTERN.match(form) for form in surface_forms):
            # Not every item carries an English label; say so instead of
            # implying the endpoint returned the wrong entity.
            note = ("the unique answer has no English label on the endpoint, so it could not "
                    "be compared with the gold string")
        else:
            note = "the endpoint returns a unique answer that does not match the gold string"
        return {"mode": "answer", "verified": matched, "checked_utc": checked,
                "endpoint": self.endpoint, "result_count": 1,
                "returned_labels": surface_forms[:5], "note": note}

    # -- abstraction -------------------------------------------------------

    def usable_descriptions(self, entity_label: str, answer: str) -> List[str]:
        """Descriptions of ``entity_label`` that do not give the answer away."""
        needle = (answer or "").strip().lower()
        out = []
        for description in self.engine.describe(entity_label):
            if needle and needle in description.lower():
                self.stats.drop("description_leaks_the_answer")
                continue
            out.append(description)
        return out

    def abstract_constraints(self, constraints: Sequence[str], answer: str) -> List[str]:
        """Replace each constraint value with an indirect description of it."""
        out: List[str] = []
        for constraint in constraints[:self.max_constraints]:
            if ": " not in constraint:
                out.append(constraint)
                continue
            relation, entity = constraint.split(": ", 1)
            descriptions = self.usable_descriptions(entity, answer)
            if descriptions:
                description = self.rng.choice(descriptions[:3])
                out.append(f"abstract [{relation}: {entity}] As [{relation}: {description}]")
            else:
                out.append(constraint)
        return out

    def build_multi_attribute(self, parent: Dict) -> Optional[Dict]:
        constraints = (parent.get("constraint_info") or {}).get("constraints") or []
        if len(constraints) < 2:
            self.stats.drop("parent_has_too_few_constraints")
            return None
        abstracted = self.abstract_constraints(constraints, str(parent.get('answer', '')))
        if sum(1 for a in abstracted if a.startswith("abstract [")) < 1:
            self.stats.drop("no_abstraction_available")
            return None
        prompt = f"""
Reframe the original question into a more challenging but still solvable version.

Original Question: {parent['question']}
Answer: {parent['answer']}

Abstract Constraints (structured form):
{'; '.join(abstracted)}

Instructions:
1. Use the abstract constraints instead of the original entities when rewriting the question.
2. Convert the structured triples into natural language phrasing.
   For example:
   - (Sinulog festival, location, Cebu City) -> "a festival held in Cebu City"
   - (Partido Demokratiko Pilipino, country, Philippines) -> "a political party in the Philippines"
3. Combine the abstract constraints naturally into a single question.
4. Make the new question more challenging than the original, but still fair and solvable.
5. Ensure the reasoning path is clear and avoid obscure metaphors.

Output: Only generate the transformed question, nothing else.
"""
        messages = [{"role": "system",
                     "content": "You create challenging questions using multiple indirect "
                                "constraints. Keep them understandable but requiring knowledge "
                                "to solve."},
                    {"role": "user", "content": prompt}]
        question = self.writer.write(messages, stub=stub_question(abstracted))
        if not question or len(question.strip()) <= 5:
            self.stats.drop("llm_failure")
            return None
        return {
            "question": question.strip(),
            "type": "abstract_multi_attribute",
            "abstraction_info": {
                "abstract_constraints": abstracted,
                "original_constraints": list(constraints),
                "abstraction_source": self.engine.source,
            },
        }

    def build_bridge(self, parent: Dict) -> Optional[Dict]:
        bridge = parent.get("bridge_info") or {}
        start = bridge.get("start_entity", "")
        intermediate = bridge.get("intermediate_entity", "")
        if not (start and intermediate):
            self.stats.drop("parent_bridge_info_incomplete")
            return None
        answer = str(parent.get("answer", ""))
        start_descriptions = self.usable_descriptions(start, answer)
        intermediate_descriptions = self.usable_descriptions(intermediate, answer)
        if not start_descriptions or not intermediate_descriptions:
            self.stats.drop("no_abstraction_available")
            return None
        start_description = self.rng.choice(start_descriptions[:3])
        intermediate_description = self.rng.choice(intermediate_descriptions[:3])
        prompt = f"""
Create a more abstract but still understandable question using these indirect descriptions:

Original: {parent['question']}
Answer: {parent['answer']}

Entity 1 ({start}) -> {start_description}
Entity 2 ({intermediate}) -> {intermediate_description}

Requirements:
1. Use indirect descriptions instead of direct names
2. Make it more challenging but still solvable
3. Keep the logical reasoning path clear
4. Avoid overly poetic language that obscures meaning

Generate only the question:"""
        messages = [{"role": "system",
                     "content": "You create challenging but solvable questions using indirect "
                                "descriptions. Focus on clarity while maintaining difficulty."},
                    {"role": "user", "content": prompt}]
        question = self.writer.write(
            messages, stub=stub_question([start_description, intermediate_description]))
        if not question or len(question.strip()) <= 5:
            self.stats.drop("llm_failure")
            return None
        return {
            "question": question.strip(),
            "type": "abstract_bridge",
            "abstraction_info": {
                "start_entity_abstraction": start_description,
                "intermediate_entity_abstraction": intermediate_description,
                "abstraction_source": self.engine.source,
            },
        }

    def run(self, parents: Sequence[Dict], target: int) -> List[Dict]:
        shuffled = self.rng.sample(list(parents), len(parents))
        produced: List[Dict] = []
        for index, parent in enumerate(shuffled, start=1):
            if len(produced) >= target:
                break
            kind = parent.get("type")
            if kind == "multi_attribute":
                built = self.build_multi_attribute(parent)
            elif kind == "bridge":
                built = self.build_bridge(parent)
            else:
                self.stats.drop(f"unsupported_parent_type:{kind}")
                continue
            if built is None:
                continue

            parent_flag = parent.get("answer_verified")
            if parent_flag is None:
                parent_flag = parent.get("verified_with_sparql")
            verification = self.verify_answer(parent.get("sparql_verification", ""),
                                              str(parent.get("answer", "")), parent_flag)
            if not verification["verified"] and not self.keep_unverified:
                self.stats.drop("answer_not_verified")
                logger.info("dropped '%s': %s", str(parent.get("answer"))[:40],
                            verification["note"])
                continue

            produced.append({
                "question": built["question"],
                "answer": parent.get("answer"),
                "level": 3,
                "type": built["type"],
                "reasoning_chain": parent.get("reasoning_chain", []),
                "sparql_verification": parent.get("sparql_verification", ""),
                "sparql_verification_source": "inherited_from_level2",
                "original_level2_question": parent.get("question", ""),
                "abstraction_info": built["abstraction_info"],
                "answer_verified": bool(verification["verified"]),
                "verification": verification,
                "question_source": self.writer.source_label,
                "generator_model": self.writer.model_label,
            })
            if index % 25 == 0:
                logger.info("Processed %d/%d parents, kept %d", index, len(shuffled), len(produced))
        return produced


def stub_question(descriptions: Sequence[str]) -> str:
    """Deterministic offline stand-in for the model call."""
    cleaned = []
    for description in descriptions:
        text = description
        if text.startswith("abstract [") and "] As [" in text:
            text = text.split("] As [", 1)[1].rstrip("]")
        cleaned.append(text.strip())
    return "Which entity is described by all of the following: " + "; ".join(cleaned) + "?"


def default_output_path(count: int, year: str) -> Path:
    return PROJECT_ROOT / "outputs" / "questions" / f"level3_{count}_questions_{year}.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Level 3 abstracted questions from Level 2 questions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=None,
                        help="Triple CSV used as a local source of entity descriptions. "
                             "When omitted, the historical default locations are tried")
    parser.add_argument("--level2", default=None,
                        help="Level 2 questions JSON (default: the newest "
                             "outputs/questions/level2_*.json)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: outputs/questions/"
                             "level3_<kept>_questions_<year>.json)")
    parser.add_argument("--report", default=None,
                        help="Run report JSON path (default: <output>.run_report.json)")
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS,
                        help="How many Level 3 questions to produce")
    parser.add_argument("--max-constraints", type=int, default=3,
                        help="How many parent constraints to abstract per question")
    parser.add_argument("--max-descriptions", type=int, default=5,
                        help="How many candidate descriptions to keep per entity")
    parser.add_argument("--abstraction-source", default="auto",
                        choices=["auto", "wikidata", "local", "reverse_hop"],
                        help="Where indirect descriptions come from: Wikidata statements, the "
                             "input CSV, a reverse hop, or all of them in that order")
    parser.add_argument("--verify", default="count", choices=["count", "answer", "inherit", "skip"],
                        help="How answer_verified is decided: 'count' re-runs the inherited "
                             "COUNT program (== 1); 'answer' additionally checks that the "
                             "endpoint returns the gold string; 'inherit' copies the parent's "
                             "flag without checking; 'skip' writes answer_verified=false")
    parser.add_argument("--keep-unverified", action="store_true",
                        help="Keep instances that failed verification (with "
                             "answer_verified=false) instead of dropping them")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for parent order and description choice")
    parser.add_argument("--year", default=None,
                        help="Release year recorded in the metadata and the default filename")
    parser.add_argument("--model", default=None,
                        help=f"Chat model used to phrase the questions "
                             f"(default: $LSB_GENERATOR_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: $OPENAI_API_KEY, then .env)")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Completion token cap for a single question")
    parser.add_argument("--endpoint", default=None,
                        help=f"SPARQL endpoint (default: $SPARQL_ENDPOINT, then "
                             f"{config.DEFAULT_SPARQL_ENDPOINT})")
    parser.add_argument("--min-interval", type=float, default=1.0,
                        help="Minimum seconds between SPARQL requests")
    parser.add_argument("--dry-run", action="store_true",
                        help="Replace the model call with a deterministic template stub "
                             "(no API key required). SPARQL verification still runs")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.num_questions <= 0:
        raise SystemExit("--num-questions must be positive")

    input_path = resolve_input(args.input)
    level2_path = resolve_level2(args.level2)
    started = datetime.now(timezone.utc)
    year = args.year or str(started.year)
    model = config.get("LSB_GENERATOR_MODEL", override=args.model, default=DEFAULT_MODEL)
    rng = random.Random(args.seed)
    random.seed(args.seed)
    stats = filters.FilterStats()

    provisional_output = Path(args.output) if args.output else default_output_path(0, year)
    ensure_parent(provisional_output)

    try:
        parents, level2_meta = load_instances(level2_path)
    except DatasetFormatError as exc:
        raise SystemExit(f"Could not read the Level 2 file: {exc}")
    stats.stage("level 2 parents", len(parents))
    logger.info("Loaded %d Level 2 questions from %s", len(parents), level2_path)

    try:
        writer = QuestionWriter(model=model, base_url=args.base_url, api_key=args.api_key,
                                dry_run=args.dry_run, temperature=args.temperature,
                                max_tokens=args.max_tokens)
    except MissingCredential as exc:
        raise SystemExit(f"{exc}\n  Or pass --dry-run to exercise the pipeline without a model.")

    keep_unverified = args.keep_unverified
    if args.verify == "skip" and not keep_unverified:
        # Nothing is checked in this mode, so dropping "unverified" items would
        # discard the whole run.
        logger.info("--verify skip keeps every instance; they are written with "
                    "answer_verified=false")
        keep_unverified = True

    needs_sparql = args.verify in ("count", "answer") or args.abstraction_source != "local"
    client = SparqlClient(endpoint=args.endpoint,
                          min_interval=args.min_interval) if needs_sparql else None

    try:
        frame = load_triples(input_path)
        engine = NodeExpansionEngine(frame, client, source=args.abstraction_source,
                                     max_properties=args.max_descriptions, stats=stats)
        generator = Level3Generator(engine=engine, writer=writer, client=client,
                                    verify=args.verify, keep_unverified=keep_unverified,
                                    max_constraints=args.max_constraints, rng=rng, stats=stats,
                                    endpoint=None if client is None else client.endpoint)
        qa_pairs = generator.run(parents, args.num_questions)
    finally:
        writer.close()

    stats.stage("questions written", len(qa_pairs))
    finished = datetime.now(timezone.utc)
    output_path = Path(args.output) if args.output else default_output_path(len(qa_pairs), year)
    ensure_parent(output_path)

    verified_count = sum(1 for qa in qa_pairs if qa["answer_verified"])
    metadata = {
        "description": "Level 3 abstracted questions derived from Level 2",
        "level": 3,
        "year": year,
        "total_questions": len(qa_pairs),
        "answer_verified_count": verified_count,
        "verification_mode": args.verify,
        "verification_method": {
            "count": "the inherited Level 2 COUNT program was re-run and returned 1",
            "answer": "the inherited Level 2 program was re-run and returned exactly the "
                      "gold answer",
            "inherit": "no check was run; the flag was copied from the Level 2 parent",
            "skip": "no check was run; answer_verified is false everywhere",
        }[args.verify],
        "sparql_verification_source": "inherited_from_level2",
        "sparql_endpoint": None if client is None else client.endpoint,
        "abstraction_source": args.abstraction_source,
        "model": writer.model_label,
        "requested_model": model,
        "question_source": writer.source_label,
        "dry_run": bool(args.dry_run),
        "seed": args.seed,
        "num_questions_requested": args.num_questions,
        "keep_unverified": bool(keep_unverified),
        "input_file": str(input_path),
        "level2_file": str(level2_path),
        "level2_metadata": level2_meta,
        "generator": "scripts/generate_level3.py",
        "generation_started_utc": started.isoformat(timespec="seconds"),
        "generation_finished_utc": finished.isoformat(timespec="seconds"),
    }
    output_path.write_text(
        json.dumps({"metadata": metadata, "qa_pairs": qa_pairs}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    report_path = Path(args.report) if args.report else output_path.with_suffix(".run_report.json")
    ensure_parent(report_path)
    report_path.write_text(json.dumps({"metadata": metadata, "funnel": stats.to_dict(),
                                       "llm_calls": writer.calls,
                                       "llm_failures": writer.failures},
                                      ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nLevel 3 generation complete: {len(qa_pairs)} questions "
          f"({verified_count} with answer_verified=true, mode '{args.verify}')")
    print(f"  questions : {output_path}")
    print(f"  run report: {report_path}")
    print()
    print(stats.render())
    if qa_pairs:
        print("\nExamples:")
        for qa in qa_pairs[:3]:
            print(f"  L3: {qa['question']}")
            print(f"  L2: {qa['original_level2_question']}")
            print(f"  A : {qa['answer']}  [verified={qa['answer_verified']}]")
    if len(qa_pairs) < args.num_questions:
        logger.warning("Produced %d of the %d requested questions; see the drop reasons above.",
                       len(qa_pairs), args.num_questions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
