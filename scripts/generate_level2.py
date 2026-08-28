#!/usr/bin/env python3
"""Level 2 (multi-constraint and bridge) question generation.

Two generators share the same triple CSV:

* ``multi_attribute`` -- grow a conjunction of (relation, value) constraints on
  one subject until the conjunction identifies exactly one Wikidata entity, and
  check that this entity really is the subject the constraints came from.
* ``bridge`` -- chain A --r1--> B --r2--> C from the input and keep the chain
  only when the two-hop program returns exactly one answer. The previous
  release built these paths with ``if True:  # Previously: if answer_count == 1``,
  i.e. with the verification disabled; here it runs, and the generator is opt-in
  via ``--include-bridge``.

Predicates are routed through :func:`livesearchbench.filters.is_allowed_relation`,
so meta predicates such as ``P31`` (instance of) can no longer become question
constraints, and counting goes through :meth:`SparqlClient.count`, which raises
on a failed request instead of returning 0 and having it read as "no match".

Examples:
    python scripts/generate_level2.py --input data/sample/triple_changes_sample.csv \
        --dry-run --num-questions 2 --output outputs/questions/demo_level2.json

    python scripts/generate_level2.py --input outputs/extracted_triples/triple_changes.csv \
        --model gpt-4o --num-questions 300 --seed 0 --include-bridge
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from livesearchbench import config, filters
from livesearchbench.config import MissingCredential
from livesearchbench.dataio import ensure_parent
from livesearchbench.http import PoliteSession, RequestFailed
from livesearchbench.sparql import SparqlClient, SparqlError

logger = logging.getLogger("generate_level2")

#: Shared default across the three generators; override with --model.
DEFAULT_MODEL = "gpt-4o"
DEFAULT_NUM_QUESTIONS = 300
DEFAULT_CANDIDATE_POOL = 2000
DEFAULT_MAX_ROWS = 30000

EXTRACTOR_COLUMNS = ("entity_id", "entity_label", "property_id", "property_label", "new_value")
TRIPLE_COLUMNS = ("subject_id", "subject_label", "predicate_id", "predicate_label",
                  "object_id", "object_label")

FALLBACK_INPUTS = (
    PROJECT_ROOT / "outputs" / "extracted_triples" / "triple_changes_latest.csv",
    PROJECT_ROOT / "data" / "final_changed_item_with_id.csv",
    PROJECT_ROOT / "data" / "sample" / "triple_changes_sample.csv",
)

#: Extra properties fetched from Wikidata when --extend-attributes is on. The
#: deny-listed IDs from filters.EXCLUDED_PROPERTY_IDS are removed below, which
#: is what drops P31 -- it was in this list in the previous release even though
#: the paper excludes it.
EXTENSION_PROPERTY_IDS: Tuple[str, ...] = tuple(
    pid for pid in (
        "P27", "P19", "P20", "P106", "P31", "P136", "P495", "P37", "P36", "P17",
        "P131", "P276", "P57", "P50", "P175", "P364", "P407", "P840", "P159",
        "P937", "P108", "P69", "P463", "P102", "P641", "P30", "P38", "P122",
        "P735", "P734", "P4552",
    ) if pid not in filters.EXCLUDED_PROPERTY_IDS
)

#: Relation pairs that make a two-hop chain vacuous (A follows B follows C, ...).
TEMPORAL_RELATIONS = frozenset({"P155", "P156", "P1365", "P1366"})
INVERSE_PAIRS = (
    ("P155", "P156"),
    ("P1365", "P1366"),
    ("P527", "P361"),
    ("P749", "P355"),
    ("P276", "P131"),
)


class QuestionWriter:
    """Chat-completions client with a deterministic offline stub.

    ``--dry-run`` returns the caller's template instead of calling a model, and
    every instance records ``question_source`` so stub output is never mistaken
    for model output.
    """

    def __init__(self, *, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 dry_run: bool = False, temperature: float = 0.8, max_tokens: int = 256,
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
            self._session = PoliteSession(component="LiveSearchBench-L2", max_attempts=max_attempts)

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
    """Return the CSV to read, honouring the historical auto-detection order."""
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
        + "\n  Produce one with scripts/extract_triple_changes.py."
    )


def load_triples(path: Path, *, max_rows: int, rng: random.Random,
                 stats: filters.FilterStats) -> pd.DataFrame:
    """Load a triple CSV in either the extractor format or the legacy format."""
    frame = pd.read_csv(path)
    stats.stage("csv rows", len(frame))
    logger.info("Loaded %d rows from %s", len(frame), path)

    if "new_value" in frame.columns:
        missing = [c for c in EXTRACTOR_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"{path} looks like extractor output but lacks columns: {missing}")
        entity_valued = frame[frame["new_value"].astype(str).str.match(r"^Q\d+$", na=False)]
        if len(entity_valued) < len(frame):
            stats.drop("object_is_not_an_entity", len(frame) - len(entity_valued))
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
    stats.stage("entity-valued triples", len(frame))
    if frame.empty:
        raise ValueError(f"{path} yielded zero usable triples after format conversion.")
    if len(frame) > max_rows:
        frame = frame.sample(n=max_rows, random_state=rng.randrange(2 ** 31))
        logger.info("Sampled the input down to %d rows (--max-rows)", max_rows)
    return frame


def filter_relations(frame: pd.DataFrame, *, use_allowlist: bool,
                     stats: filters.FilterStats) -> pd.DataFrame:
    """Drop rows whose predicate or subject the paper's filters exclude."""
    mask: List[bool] = []
    for row in frame.itertuples(index=False):
        pid = str(row.predicate_id).upper()
        if pid in filters.EXCLUDED_PROPERTY_IDS:
            stats.drop(f"excluded_property:{pid}")
            mask.append(False)
            continue
        if not filters.is_allowed_relation(pid, str(row.predicate_label),
                                           use_allowlist=use_allowlist):
            stats.drop("relation_not_in_allowlist")
            mask.append(False)
            continue
        ok, reason = filters.is_allowed_entity(label=str(row.subject_label), require_enwiki=False)
        if not ok:
            stats.drop(f"subject_rejected:{reason}")
            mask.append(False)
            continue
        mask.append(True)
    kept = frame[mask]
    stats.stage("allowed relations", len(kept))
    if kept.empty:
        raise ValueError(
            "No triple survived relation filtering.\n"
            "  Pass --no-relation-allowlist to keep every non-excluded predicate."
        )
    return kept


def index_attributes(frame: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
    """subject_id -> list of {predicate_id, predicate_label, object_id, object_label}."""
    index: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    seen: set = set()
    for row in frame.itertuples(index=False):
        key = filters.dedup_key(str(row.subject_id), str(row.predicate_id)) + f"|{row.object_id}"
        if key in seen:
            continue
        seen.add(key)
        index[str(row.subject_id)].append({
            "predicate_id": str(row.predicate_id),
            "predicate_label": str(row.predicate_label),
            "object_id": str(row.object_id),
            "object_label": str(row.object_label),
        })
    return index


def entity_labels(frame: pd.DataFrame) -> Dict[str, str]:
    """Best-effort id -> label map built from both ends of every triple."""
    labels: Dict[str, str] = {}
    for row in frame.itertuples(index=False):
        labels.setdefault(str(row.subject_id), str(row.subject_label))
        labels.setdefault(str(row.object_id), str(row.object_label))
    return labels


def constraint_where(attributes: Sequence[Dict[str, str]]) -> str:
    return " ".join(f"?entity wdt:{a['predicate_id']} wd:{a['object_id']} ."
                    for a in attributes)


def count_query(attributes: Sequence[Dict[str, str]]) -> str:
    """The COUNT program stored in ``sparql_verification``."""
    return (f"SELECT (COUNT(?entity) AS ?count) WHERE {{\n"
            f"  {constraint_where(attributes)}\n"
            f"}}")


def matching_entities(client: SparqlClient, attributes: Sequence[Dict[str, str]],
                      limit: int = 2) -> List[str]:
    """Return up to ``limit`` QIDs satisfying the conjunction.

    Returning the identifiers rather than only a count lets the caller confirm
    that the unique match is the entity the constraints were taken from; a count
    of one on its own does not prove that.
    """
    query = (f"SELECT DISTINCT ?entity WHERE {{\n  {constraint_where(attributes)}\n}}\n"
             f"LIMIT {limit}")
    data = client.query(query)
    out = []
    for row in data.get("results", {}).get("bindings", []):
        value = row.get("entity", {}).get("value", "")
        if value:
            out.append(value.rsplit("/", 1)[-1])
    return out


class MultiAttributeGenerator:
    """Grow constraint conjunctions until they pin down exactly one entity."""

    def __init__(self, frame: pd.DataFrame, client: SparqlClient, *, rng: random.Random,
                 stats: filters.FilterStats, min_constraints: int, max_constraints: int,
                 attempts: int, extend: bool, use_allowlist: bool,
                 max_consecutive_errors: int, candidate_pool: int) -> None:
        self.frame = frame
        self.client = client
        self.rng = rng
        self.stats = stats
        self.min_constraints = min_constraints
        self.max_constraints = max_constraints
        self.attempts = attempts
        self.extend = extend
        self.use_allowlist = use_allowlist
        self.max_consecutive_errors = max_consecutive_errors
        self.candidate_pool = candidate_pool
        self.attributes = index_attributes(frame)
        self.labels = entity_labels(frame)
        self._consecutive_errors = 0

    def _note_error(self, message: str) -> None:
        self._consecutive_errors += 1
        self.stats.drop("sparql_error")
        logger.warning("%s", message)
        if self._consecutive_errors >= self.max_consecutive_errors:
            raise SystemExit(
                f"Aborting: {self._consecutive_errors} consecutive SPARQL failures against "
                f"{self.client.endpoint}.\n  Check connectivity or raise --max-sparql-errors."
            )

    def find_entities(self, target: int) -> List[Dict]:
        """Return at most ``target`` entities with a verified constraint set."""
        ordered = sorted(self.attributes.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        candidates = [(eid, attrs) for eid, attrs in ordered if len(attrs) >= self.min_constraints]
        self.stats.stage("multi-attribute: subjects with enough attributes", len(candidates))
        candidates = candidates[:self.candidate_pool]
        self.stats.stage("multi-attribute: subjects examined", len(candidates))
        for entity_id, attrs in ordered:
            if len(attrs) < self.min_constraints:
                self.stats.drop("too_few_attributes")

        found: List[Dict] = []
        for entity_id, attrs in candidates:
            if len(found) >= target:
                break
            constraints = self.find_constraints(entity_id, attrs)
            if constraints is None:
                continue
            found.append({
                "entity_id": entity_id,
                "entity_label": self.labels.get(entity_id, entity_id),
                "constraints": constraints,
                "total_attributes": len(attrs),
            })
            logger.info("constraint set found for %s (%d constraints)",
                        self.labels.get(entity_id, entity_id), constraints["constraint_count"])
        self.stats.stage("multi-attribute: constraint sets verified", len(found))
        return found

    def find_constraints(self, entity_id: str, attributes: Sequence[Dict[str, str]]) -> Optional[Dict]:
        """Search for a conjunction that matches ``entity_id`` and nothing else."""
        pool = list(attributes)
        if self.extend:
            pool = self.extended_attributes(entity_id, pool)

        for attempt in range(max(1, self.attempts)):
            shuffled = self.rng.sample(pool, len(pool))
            upper = min(self.max_constraints, len(shuffled))
            for size in range(self.min_constraints, upper + 1):
                selected = shuffled[:size]
                try:
                    matches = matching_entities(self.client, selected)
                except SparqlError as exc:
                    self._note_error(f"constraint query failed for {entity_id}: {exc}")
                    break
                self._consecutive_errors = 0
                if not matches:
                    # Monotone: adding constraints cannot bring matches back.
                    self.stats.drop("constraints_match_nothing")
                    break
                if len(matches) > 1:
                    continue
                if matches[0] != entity_id:
                    self.stats.drop("unique_match_is_a_different_entity")
                    continue
                return {
                    "attributes": list(selected),
                    "sparql_query": count_query(selected),
                    "constraint_count": size,
                    "attempt": attempt + 1,
                }
            else:
                self.stats.drop("not_unique_within_constraint_budget")
        return None

    def extended_attributes(self, entity_id: str,
                            existing: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        """Fetch extra allowed attributes for the entity from Wikidata."""
        known = {a["predicate_id"] for a in existing}
        wanted = [pid for pid in EXTENSION_PROPERTY_IDS if pid not in known]
        if not wanted:
            return list(existing)
        values = " ".join(f"wd:{pid}" for pid in wanted)
        query = f"""
SELECT ?prop ?propLabel ?value ?valueLabel WHERE {{
  VALUES ?prop {{ {values} }}
  ?prop wikibase:directClaim ?property .
  wd:{entity_id} ?property ?value .
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT 50
"""
        try:
            data = self.client.query(query)
        except SparqlError as exc:
            self._note_error(f"attribute extension failed for {entity_id}: {exc}")
            return list(existing)
        self._consecutive_errors = 0

        out = list(existing)
        for row in data.get("results", {}).get("bindings", []):
            property_uri = row.get("prop", {}).get("value", "")
            value_uri = row.get("value", {}).get("value", "")
            property_label = row.get("propLabel", {}).get("value", "")
            value_label = row.get("valueLabel", {}).get("value", "")
            if not property_uri or not value_uri:
                continue
            property_id = property_uri.rsplit("/", 1)[-1]
            object_id = value_uri.rsplit("/", 1)[-1]
            if not property_id.startswith("P") or not object_id.startswith("Q"):
                continue
            if not filters.is_allowed_relation(property_id, property_label,
                                               use_allowlist=self.use_allowlist):
                self.stats.drop("extension_relation_not_allowed")
                continue
            if not value_label or value_label == object_id:
                continue
            out.append({"predicate_id": property_id,
                        "predicate_label": property_label or property_id,
                        "object_id": object_id,
                        "object_label": value_label})
        if len(out) > len(existing):
            logger.debug("extended %s with %d attributes", entity_id, len(out) - len(existing))
        return out


def stub_multi_attribute_question(entity: Dict) -> str:
    """Deterministic offline stand-in for the model call."""
    parts = [f"{a['predicate_label']} {a['object_label']}"
             for a in entity["constraints"]["attributes"]]
    return "Which entity has " + " and ".join(parts) + "?"


def multi_attribute_prompt(entity: Dict) -> List[Dict[str, str]]:
    described = " and ".join(f"has {a['predicate_label']} {a['object_label']}"
                             for a in entity["constraints"]["attributes"])
    prompt = f"""
Generate a natural question asking about an entity with multiple specific constraints:

Entity: {entity['entity_label']}
Constraints: The entity {described}

The question should ask "Which entity..." or "What..." and describe these constraints naturally.

Answer: {entity['entity_label']}

Examples:
- "Which country has Paris as its capital and French as its official language?"
- "What company was founded by Steve Jobs and is headquartered in Cupertino?"
- "Which university is located in Cambridge and was founded in 1209?"

Make it flow naturally and be specific. Generate only the question:"""
    return [{"role": "system",
             "content": "You generate natural multi-constraint questions that require reasoning "
                        "through multiple attributes."},
            {"role": "user", "content": prompt}]


def generate_multi_attribute_questions(entities: Sequence[Dict], writer: QuestionWriter, *,
                                       stats: filters.FilterStats) -> List[Dict]:
    qa_pairs: List[Dict] = []
    for entity in entities:
        constraints = entity["constraints"]
        attributes = constraints["attributes"]
        question = writer.write(multi_attribute_prompt(entity),
                                stub=stub_multi_attribute_question(entity))
        if not question or len(question.strip()) <= 5:
            stats.drop("llm_failure")
            logger.warning("no question for %s", entity["entity_label"])
            continue
        qa_pairs.append({
            "question": question.strip(),
            "answer": entity["entity_label"],
            "level": 2,
            "type": "multi_attribute",
            "reasoning_chain": [[entity["entity_label"], a["predicate_label"], a["object_label"]]
                                for a in attributes],
            "sparql_verification": constraints["sparql_query"],
            "verified_with_sparql": True,
            "question_source": writer.source_label,
            "generator_model": writer.model_label,
            "constraint_info": {
                "constraint_count": constraints["constraint_count"],
                "total_attributes": entity["total_attributes"],
                "constraints": [f"{a['predicate_label']}: {a['object_label']}" for a in attributes],
            },
            "source_entity": {"entity_id": entity["entity_id"],
                              "entity_label": entity["entity_label"]},
        })
    return qa_pairs


def is_meaningless_bridge(relation_ab: str, relation_bc: str) -> bool:
    """Reject chains whose two hops cancel out or are pure sequence links."""
    if relation_ab in TEMPORAL_RELATIONS and relation_bc in TEMPORAL_RELATIONS:
        return True
    for first, second in INVERSE_PAIRS:
        if {relation_ab, relation_bc} == {first, second}:
            return True
    return relation_ab == relation_bc


def bridge_count_query(entity_a: str, relation_ab: str, relation_bc: str) -> str:
    return (f"SELECT (COUNT(DISTINCT ?answer) AS ?count) WHERE {{\n"
            f"  wd:{entity_a} wdt:{relation_ab} ?intermediate .\n"
            f"  ?intermediate wdt:{relation_bc} ?answer .\n"
            f"}}")


def find_bridge_paths(frame: pd.DataFrame, client: SparqlClient, *, max_paths: int,
                      rng: random.Random, stats: filters.FilterStats,
                      max_consecutive_errors: int) -> List[Dict]:
    """Join the triple table on the intermediate entity and verify each chain."""
    by_subject: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        by_subject[str(row.subject_id)].append({
            "predicate_id": str(row.predicate_id),
            "predicate_label": str(row.predicate_label),
            "object_id": str(row.object_id),
            "object_label": str(row.object_label),
            "subject_label": str(row.subject_label),
        })

    chains: List[Tuple[str, Dict[str, str], Dict[str, str]]] = []
    for entity_a, first_hops in by_subject.items():
        for first in first_hops:
            for second in by_subject.get(first["object_id"], []):
                if second["object_id"] in (entity_a, first["object_id"]):
                    continue
                if is_meaningless_bridge(first["predicate_id"], second["predicate_id"]):
                    stats.drop("meaningless_bridge")
                    continue
                chains.append((entity_a, first, second))
    stats.stage("bridge: candidate chains", len(chains))
    rng.shuffle(chains)

    verified: List[Dict] = []
    used: set = set()
    consecutive_errors = 0
    for entity_a, first, second in chains:
        if len(verified) >= max_paths:
            break
        if used & {entity_a, first["object_id"], second["object_id"]}:
            stats.drop("bridge_entity_already_used")
            continue
        query = bridge_count_query(entity_a, first["predicate_id"], second["predicate_id"])
        try:
            count = client.count(query)
        except SparqlError as exc:
            consecutive_errors += 1
            stats.drop("sparql_error")
            logger.warning("bridge verification failed for %s: %s", entity_a, exc)
            if consecutive_errors >= max_consecutive_errors:
                raise SystemExit(
                    f"Aborting: {consecutive_errors} consecutive SPARQL failures against "
                    f"{client.endpoint}."
                )
            continue
        consecutive_errors = 0
        if count != 1:
            stats.drop("bridge_answer_not_unique")
            continue
        verified.append({
            "entity_a": entity_a,
            "entity_a_label": first["subject_label"],
            "relation_ab_id": first["predicate_id"],
            "relation_ab_label": first["predicate_label"],
            "entity_b": first["object_id"],
            "entity_b_label": first["object_label"],
            "relation_bc_id": second["predicate_id"],
            "relation_bc_label": second["predicate_label"],
            "entity_c": second["object_id"],
            "entity_c_label": second["object_label"],
            "sparql_query": query,
        })
        used.update({entity_a, first["object_id"], second["object_id"]})
    stats.stage("bridge: chains verified", len(verified))
    return verified


def stub_bridge_question(bridge: Dict) -> str:
    return (f"What is the {bridge['relation_bc_label']} of the "
            f"{bridge['relation_ab_label']} of {bridge['entity_a_label']}?")


def bridge_prompt(bridge: Dict) -> List[Dict[str, str]]:
    prompt = f"""
Generate a natural two-hop question that asks about the final destination in this path:

Path: {bridge['entity_a_label']} --{bridge['relation_ab_label']}--> {bridge['entity_b_label']} --{bridge['relation_bc_label']}--> {bridge['entity_c_label']}

The question should ask: "What is the {bridge['relation_bc_label']} of the {bridge['relation_ab_label']} of {bridge['entity_a_label']}?"

But make it more natural and conversational. The answer should be: {bridge['entity_c_label']}

Examples:
- "Where was the director of Inception born?"
- "What genre does the author of Harry Potter write?"
- "Which country is the capital of France located in?"

Generate only the question:"""
    return [{"role": "system",
             "content": "You generate natural multi-hop questions that require reasoning through "
                        "intermediate entities."},
            {"role": "user", "content": prompt}]


def generate_bridge_questions(bridges: Sequence[Dict], writer: QuestionWriter, *,
                              stats: filters.FilterStats) -> List[Dict]:
    qa_pairs: List[Dict] = []
    for bridge in bridges:
        question = writer.write(bridge_prompt(bridge), stub=stub_bridge_question(bridge))
        if not question or len(question.strip()) <= 5:
            stats.drop("llm_failure")
            continue
        qa_pairs.append({
            "question": question.strip(),
            "answer": bridge["entity_c_label"],
            "level": 2,
            "type": "bridge",
            "reasoning_chain": [
                [bridge["entity_a_label"], bridge["relation_ab_label"], bridge["entity_b_label"]],
                [bridge["entity_b_label"], bridge["relation_bc_label"], bridge["entity_c_label"]],
            ],
            "sparql_verification": bridge["sparql_query"],
            "verified_with_sparql": True,
            "question_source": writer.source_label,
            "generator_model": writer.model_label,
            "bridge_info": {
                "start_entity": bridge["entity_a_label"],
                "intermediate_entity": bridge["entity_b_label"],
                "final_entity": bridge["entity_c_label"],
                "intermediate_count": 1,
            },
        })
    return qa_pairs


def default_output_path(count: int, year: str) -> Path:
    return PROJECT_ROOT / "outputs" / "questions" / f"level2_{count}_questions_{year}.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Level 2 multi-constraint (and optional bridge) questions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=None,
                        help="Triple CSV: extractor output (11 columns incl. new_value) or a "
                             "legacy subject/predicate/object CSV. When omitted, the historical "
                             "default locations are tried in order")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: outputs/questions/"
                             "level2_<kept>_questions_<year>.json)")
    parser.add_argument("--report", default=None,
                        help="Run report JSON path (default: <output>.run_report.json)")
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS,
                        help="How many verified questions to produce in total")
    parser.add_argument("--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL,
                        help="Cap on how many subjects are examined for constraint sets")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                        help="Sub-sample the input CSV to at most this many rows")
    parser.add_argument("--min-constraints", type=int, default=2,
                        help="Smallest conjunction size to test")
    parser.add_argument("--max-constraints", type=int, default=5,
                        help="Largest conjunction size to test")
    parser.add_argument("--constraint-attempts", type=int, default=3,
                        help="Random attribute orderings tried per subject")
    parser.add_argument("--extend-attributes", action="store_true",
                        help="Fetch extra attributes from Wikidata when the CSV alone cannot "
                             "pin down the entity (they are filtered the same way)")
    parser.add_argument("--include-bridge", action="store_true",
                        help="Also generate two-hop bridge questions (verified, unlike the "
                             "previous release)")
    parser.add_argument("--max-bridges", type=int, default=50,
                        help="Cap on verified bridge chains when --include-bridge is set")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for attribute ordering (recorded in the metadata)")
    parser.add_argument("--year", default=None,
                        help="Release year recorded in the metadata and the default filename")
    parser.add_argument("--model", default=None,
                        help=f"Chat model used to phrase the questions "
                             f"(default: $LSB_GENERATOR_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: $OPENAI_API_KEY, then .env)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Completion token cap for a single question")
    parser.add_argument("--endpoint", default=None,
                        help=f"SPARQL endpoint (default: $SPARQL_ENDPOINT, then "
                             f"{config.DEFAULT_SPARQL_ENDPOINT})")
    parser.add_argument("--min-interval", type=float, default=1.0,
                        help="Minimum seconds between SPARQL requests")
    parser.add_argument("--max-sparql-errors", type=int, default=20,
                        help="Abort after this many consecutive SPARQL failures")
    parser.add_argument("--no-relation-allowlist", action="store_true",
                        help="Keep every predicate that is not on the exclusion list, instead "
                             "of only the 198 allow-listed relation labels")
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
    if args.min_constraints < 1 or args.max_constraints < args.min_constraints:
        raise SystemExit("--min-constraints must be >= 1 and <= --max-constraints")

    input_path = resolve_input(args.input)
    started = datetime.now(timezone.utc)
    year = args.year or str(started.year)
    model = config.get("LSB_GENERATOR_MODEL", override=args.model, default=DEFAULT_MODEL)
    rng = random.Random(args.seed)
    random.seed(args.seed)
    stats = filters.FilterStats()

    provisional_output = Path(args.output) if args.output else default_output_path(0, year)
    ensure_parent(provisional_output)

    try:
        writer = QuestionWriter(model=model, base_url=args.base_url, api_key=args.api_key,
                                dry_run=args.dry_run, temperature=args.temperature,
                                max_tokens=args.max_tokens)
    except MissingCredential as exc:
        raise SystemExit(f"{exc}\n  Or pass --dry-run to exercise the pipeline without a model.")

    client = SparqlClient(endpoint=args.endpoint, min_interval=args.min_interval)
    try:
        frame = load_triples(input_path, max_rows=args.max_rows, rng=rng, stats=stats)
        frame = filter_relations(frame, use_allowlist=not args.no_relation_allowlist, stats=stats)

        qa_pairs: List[Dict] = []
        bridges: List[Dict] = []
        if args.include_bridge:
            bridges = find_bridge_paths(frame, client, max_paths=min(args.max_bridges,
                                                                     args.num_questions),
                                        rng=rng, stats=stats,
                                        max_consecutive_errors=args.max_sparql_errors)
            qa_pairs.extend(generate_bridge_questions(bridges, writer, stats=stats))

        remaining = max(0, args.num_questions - len(qa_pairs))
        generator = MultiAttributeGenerator(
            frame, client, rng=rng, stats=stats,
            min_constraints=args.min_constraints, max_constraints=args.max_constraints,
            attempts=args.constraint_attempts, extend=args.extend_attributes,
            use_allowlist=not args.no_relation_allowlist,
            max_consecutive_errors=args.max_sparql_errors,
            candidate_pool=args.candidate_pool)
        entities = generator.find_entities(remaining)
        qa_pairs.extend(generate_multi_attribute_questions(entities, writer, stats=stats))
    finally:
        writer.close()

    stats.stage("questions written", len(qa_pairs))
    finished = datetime.now(timezone.utc)
    output_path = Path(args.output) if args.output else default_output_path(len(qa_pairs), year)
    ensure_parent(output_path)

    multi_attribute_count = sum(1 for qa in qa_pairs if qa["type"] == "multi_attribute")
    metadata = {
        "description": "Level 2 multi-constraint questions with SPARQL-verified unique answers",
        "level": 2,
        "year": year,
        "total_questions": len(qa_pairs),
        "multi_attribute_questions": multi_attribute_count,
        "bridge_questions": len(qa_pairs) - multi_attribute_count,
        "verification_method": "SPARQL: the constraint conjunction matches exactly the source "
                               "entity (bridge chains: COUNT(DISTINCT ?answer) == 1)",
        "sparql_endpoint": client.endpoint,
        "model": writer.model_label,
        "requested_model": model,
        "question_source": writer.source_label,
        "dry_run": bool(args.dry_run),
        "seed": args.seed,
        "min_constraints": args.min_constraints,
        "max_constraints": args.max_constraints,
        "constraint_attempts": args.constraint_attempts,
        "extend_attributes": bool(args.extend_attributes),
        "include_bridge": bool(args.include_bridge),
        "candidate_pool": args.candidate_pool,
        "num_questions_requested": args.num_questions,
        "relation_allowlist": not args.no_relation_allowlist,
        "input_file": str(input_path),
        "generator": "scripts/generate_level2.py",
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

    print(f"\nLevel 2 generation complete: {len(qa_pairs)} questions "
          f"({multi_attribute_count} multi-attribute, "
          f"{len(qa_pairs) - multi_attribute_count} bridge)")
    print(f"  questions : {output_path}")
    print(f"  run report: {report_path}")
    print()
    print(stats.render())
    if qa_pairs:
        print("\nExamples:")
        for qa in qa_pairs[:3]:
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer']}  [{qa['type']}]")
    if len(qa_pairs) < args.num_questions:
        logger.warning("Produced %d of the %d requested questions; see the drop reasons above.",
                       len(qa_pairs), args.num_questions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
