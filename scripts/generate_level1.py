#!/usr/bin/env python3
"""Level 1 (single-hop) question generation.

Reads the CSV written by ``scripts/extract_triple_changes.py``, keeps the
(subject, relation, object) triples whose relation survives the paper's
relation filters, verifies with a SPARQL COUNT program that the relation has
exactly one value for that subject, and turns each surviving triple into a
natural-language question.

Everything that used to be a module-level literal is now a flag: the model, the
credentials, the SPARQL endpoint, the random seed, the candidate pool size, the
number of questions and the output path. The values that were actually used are
recorded in the output metadata, so a released file cannot misreport how it was
made.

Examples:
    # Offline smoke test: no API key needed, questions come from a template stub.
    python scripts/generate_level1.py --input data/sample/triple_changes_sample.csv \
        --dry-run --num-questions 5 --output outputs/questions/demo_level1.json

    # Real run.
    python scripts/generate_level1.py --input outputs/extracted_triples/triple_changes.csv \
        --model gpt-4o --num-questions 300 --seed 0
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
from typing import Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from livesearchbench import config, filters
from livesearchbench.config import MissingCredential
from livesearchbench.dataio import ensure_parent
from livesearchbench.http import PoliteSession, RequestFailed
from livesearchbench.sparql import SparqlClient, SparqlError

logger = logging.getLogger("generate_level1")

#: Shared default across the three generators; override with --model or
#: LSB_GENERATOR_MODEL. The previous release hardcoded a different model in
#: every script and recorded none of them.
DEFAULT_MODEL = "gpt-4o"
DEFAULT_NUM_QUESTIONS = 300
DEFAULT_CANDIDATE_POOL = 2000
DEFAULT_MAX_ROWS = 30000

#: Columns produced by scripts/extract_triple_changes.py.
EXTRACTOR_COLUMNS = ("entity_id", "entity_label", "property_id", "property_label", "new_value")
#: Columns of the normalised internal frame.
TRIPLE_COLUMNS = ("subject_id", "subject_label", "predicate_id", "predicate_label",
                  "object_id", "object_label")

SYSTEM_PROMPT = "You generate natural questions from knowledge triples."


#: Where a triple CSV is looked for when --input is omitted, in order.
FALLBACK_INPUTS = (
    PROJECT_ROOT / "outputs" / "extracted_triples" / "triple_changes_latest.csv",
    PROJECT_ROOT / "data" / "final_changed_item_with_id.csv",
    PROJECT_ROOT / "data" / "sample" / "triple_changes_sample.csv",
)


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


class QuestionWriter:
    """Chat-completions client with a deterministic offline stub.

    In ``--dry-run`` mode no HTTP call is made and the caller-supplied template
    string is returned instead, so the full pipeline can be exercised without
    credentials. Items produced that way are tagged ``question_source:
    template_stub`` so they can never be mistaken for model output.
    """

    def __init__(self, *, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 dry_run: bool = False, temperature: float = 0.7, max_tokens: int = 256,
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
            self._session = PoliteSession(component="LiveSearchBench-L1", max_attempts=max_attempts)

    @property
    def source_label(self) -> str:
        """What actually wrote the questions, for the output metadata."""
        return "template_stub" if self.dry_run else "llm"

    @property
    def model_label(self) -> str:
        return "template-stub (dry run)" if self.dry_run else self.model

    def write(self, messages: Sequence[Dict[str, str]], *, stub: str) -> Optional[str]:
        """Return the generated question, or ``None`` when generation failed."""
        self.calls += 1
        if self.dry_run:
            return stub
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
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


def load_triples(path: Path, *, max_rows: int, rng: random.Random,
                 stats: filters.FilterStats) -> pd.DataFrame:
    """Load a triple CSV in either the extractor format or the legacy format."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            f"  Produce one with scripts/extract_triple_changes.py, or use the\n"
            f"  bundled fixture data/sample/triple_changes_sample.csv."
        )
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

    frame = frame.dropna(subset=["subject_id", "predicate_id", "object_id"])
    frame = frame.astype(str)
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
            "  The allow-list holds 198 relation labels (livesearchbench/data/"
            "relation_allowlist.json).\n"
            "  Pass --no-relation-allowlist to keep every non-excluded predicate."
        )
    return kept


def prepare_candidates(frame: pd.DataFrame, *, pool_size: int, rng: random.Random,
                       stats: filters.FilterStats) -> List[Dict[str, str]]:
    """Sample a candidate pool, spread across relations, without reusing entities."""
    by_relation: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        by_relation[str(row.predicate_id)].append({
            "subject_id": str(row.subject_id),
            "subject_label": str(row.subject_label),
            "predicate_id": str(row.predicate_id),
            "predicate_label": str(row.predicate_label),
            "object_id": str(row.object_id),
            "object_label": str(row.object_label),
        })
    ordered = sorted(by_relation.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    logger.info("Found %d distinct relations", len(ordered))

    per_relation = max(1, pool_size // max(1, len(ordered)))
    candidates: List[Dict[str, str]] = []
    used_entities: set = set()
    seen_keys: set = set()

    # Two passes: the first spreads the budget evenly over relations, the second
    # tops the pool up from whatever is left.
    for take_all in (False, True):
        for _, rows in ordered:
            if len(candidates) >= pool_size:
                break
            locally_unique: Dict[str, List[Dict[str, str]]] = defaultdict(list)
            for row in rows:
                locally_unique[row["subject_id"]].append(row)
            eligible = []
            for subject_id, subject_rows in locally_unique.items():
                key = filters.dedup_key(subject_id, subject_rows[0]["predicate_id"])
                if key in seen_keys:
                    continue
                if len({r["object_id"] for r in subject_rows}) != 1:
                    stats.drop("multiple_objects_in_input")
                    seen_keys.add(key)
                    continue
                if subject_id in used_entities:
                    stats.drop("entity_already_used")
                    continue
                eligible.append(subject_rows[0])
            if not eligible:
                continue
            budget = len(eligible) if take_all else min(per_relation, len(eligible))
            budget = min(budget, pool_size - len(candidates))
            for row in rng.sample(eligible, budget):
                candidates.append(row)
                seen_keys.add(filters.dedup_key(row["subject_id"], row["predicate_id"]))
                used_entities.add(row["subject_id"])
                used_entities.add(row["object_id"])
        if len(candidates) >= pool_size:
            break

    rng.shuffle(candidates)
    stats.stage("candidate pool", len(candidates))
    logger.info("Prepared %d candidates over %d distinct subjects",
                len(candidates), len({c['subject_id'] for c in candidates}))
    return candidates


def build_verification_query(candidate: Dict[str, str]) -> str:
    """The COUNT program shipped with every released instance.

    Identifiers are validated rather than interpolated blindly: a malformed CSV
    cell would otherwise be spliced into the query string.
    """
    subject = filters.require_qid(candidate["subject_id"], field="subject_id")
    predicate = filters.require_pid(candidate["predicate_id"], field="predicate_id")
    return (f"SELECT (COUNT(?object) AS ?count) WHERE {{\n"
            f"  wd:{subject} wdt:{predicate} ?object .\n"
            f"}}")


def build_answer_query(candidate: Dict[str, str]) -> str:
    """SELECT the single object so it can be compared with the recorded gold."""
    subject = filters.require_qid(candidate["subject_id"], field="subject_id")
    predicate = filters.require_pid(candidate["predicate_id"], field="predicate_id")
    return (f"SELECT ?object WHERE {{\n"
            f"  wd:{subject} wdt:{predicate} ?object .\n"
            f"}} LIMIT 2")


def verify_candidates(candidates: Sequence[Dict[str, str]], client: Optional[SparqlClient], *,
                      target: int, stats: filters.FilterStats,
                      max_consecutive_errors: int) -> List[Dict[str, str]]:
    """Keep candidates whose relation has exactly one value on live Wikidata.

    A failed request is counted as ``sparql_error`` and never as "no results";
    too many in a row aborts the run rather than quietly producing a short file.
    """
    verified: List[Dict[str, str]] = []
    consecutive_errors = 0

    for index, candidate in enumerate(candidates, start=1):
        if len(verified) >= target:
            break
        try:
            query = build_verification_query(candidate)
        except filters.InvalidIdentifier as exc:
            # One malformed CSV row must not abort a long run.
            stats.drop("invalid_identifier")
            logger.warning("Skipping row with %s", exc)
            continue
        if client is None:
            candidate = dict(candidate, sparql_query=query, verified=False, answer_count=None)
            verified.append(candidate)
            continue
        try:
            count = client.count(query)
        except SparqlError as exc:
            consecutive_errors += 1
            stats.drop("sparql_error")
            logger.warning("SPARQL verification failed for %s/%s: %s",
                           candidate["subject_id"], candidate["predicate_id"], exc)
            if consecutive_errors >= max_consecutive_errors:
                raise SystemExit(
                    f"Aborting: {consecutive_errors} consecutive SPARQL failures against "
                    f"{client.endpoint}.\n  Check connectivity or raise --max-sparql-errors."
                )
            continue
        consecutive_errors = 0
        if count == 1:
            # Uniqueness alone is not verification: the single value Wikidata
            # returns now must also be the answer recorded in the CSV. Without
            # this, a fact that changed upstream keeps its stale gold answer and
            # is still marked verified.
            try:
                returned = client.select_labels(build_answer_query(candidate))
            except SparqlError as exc:
                stats.drop("sparql_error")
                logger.warning("Answer lookup failed for %s/%s: %s",
                               candidate["subject_id"], candidate["predicate_id"], exc)
                continue
            expected = str(candidate.get("object_id") or "").strip()
            returned_ids = [r.rsplit("/", 1)[-1] for r in returned]
            if expected and expected not in returned_ids:
                stats.drop("answer_changed_upstream")
                logger.debug("answer drift: %s/%s recorded %s, endpoint returns %s",
                             candidate["subject_id"], candidate["predicate_id"],
                             expected, returned_ids)
                continue
            verified.append(dict(candidate, sparql_query=query, verified=True,
                                 answer_count=1, answer_confirmed=bool(expected)))
            logger.debug("unique: %s --%s--> %s", candidate["subject_label"],
                         candidate["predicate_label"], candidate["object_label"])
        else:
            stats.drop("not_unique")
            logger.debug("not unique (%d values): %s --%s-->", count,
                         candidate["subject_label"], candidate["predicate_label"])
        if index % 25 == 0:
            logger.info("Verified %d/%d candidates, kept %d", index, len(candidates), len(verified))

    stats.stage("uniqueness verified", len(verified))
    return verified


def stub_question(triple: Dict[str, str]) -> str:
    """Deterministic offline stand-in for the model call."""
    return f"What is the {triple['predicate_label']} of {triple['subject_label']}?"


def build_prompt(triple: Dict[str, str]) -> List[Dict[str, str]]:
    prompt = f"""
Generate a natural, clear question based on this knowledge triple:

Subject: {triple['subject_label']}
Relation: {triple['predicate_label']}
Answer: {triple['object_label']}

Requirements:
1. Make it conversational and natural
2. The answer should be exactly "{triple['object_label']}"
3. Ask about the relationship in a human-friendly way
4. Keep it concise and clear

Examples:
- "What is the capital of France?" -> "Paris"
- "Who directed Inception?" -> "Christopher Nolan"
- "Where was Einstein born?" -> "Germany"

Generate only the question:"""
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}]


def generate_questions(triples: Sequence[Dict[str, str]], writer: QuestionWriter, *,
                       stats: filters.FilterStats) -> List[Dict]:
    """Write one question per verified triple."""
    qa_pairs: List[Dict] = []
    for index, triple in enumerate(triples, start=1):
        question = writer.write(build_prompt(triple), stub=stub_question(triple))
        if not question or len(question.strip()) <= 5:
            stats.drop("llm_failure")
            logger.warning("no question for %s --%s--> %s", triple["subject_label"],
                           triple["predicate_label"], triple["object_label"])
            continue
        qa_pairs.append({
            "question": question.strip(),
            "answer": triple["object_label"],
            "level": 1,
            "type": "single_hop",
            "reasoning_chain": [[triple["subject_label"], triple["predicate_label"],
                                 triple["object_label"]]],
            "sparql_verification": triple["sparql_query"],
            "verified_with_sparql": bool(triple["verified"]),
            "question_source": writer.source_label,
            "generator_model": writer.model_label,
            "source_triple": {
                "subject_id": triple["subject_id"],
                "subject_label": triple["subject_label"],
                "predicate_id": triple["predicate_id"],
                "predicate_label": triple["predicate_label"],
                "object_id": triple["object_id"],
                "object_label": triple["object_label"],
            },
        })
        if index % 25 == 0:
            logger.info("Wrote %d/%d questions", len(qa_pairs), len(triples))
    stats.stage("questions written", len(qa_pairs))
    return qa_pairs


def default_output_path(count: int, year: str) -> Path:
    """Derive the filename from what was really produced, not from a literal."""
    return PROJECT_ROOT / "outputs" / "questions" / f"level1_{count}_questions_{year}.json"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Level 1 single-hop questions from a Wikidata triple-change CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=None,
                        help="Triple CSV: extractor output (11 columns incl. new_value) "
                             "or a legacy subject/predicate/object CSV. When omitted, the "
                             "historical default locations are tried in order")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: outputs/questions/"
                             "level1_<kept>_questions_<year>.json)")
    parser.add_argument("--report", default=None,
                        help="Run report JSON path (default: <output>.run_report.json)")
    parser.add_argument("--num-questions", type=int, default=DEFAULT_NUM_QUESTIONS,
                        help="How many verified questions to produce")
    parser.add_argument("--candidate-pool", type=int, default=DEFAULT_CANDIDATE_POOL,
                        help="How many candidate triples to sample before verification")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS,
                        help="Sub-sample the input CSV to at most this many rows")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for candidate sampling (recorded in the metadata)")
    parser.add_argument("--year", default=None,
                        help="Release year recorded in the metadata and the default filename "
                             "(default: the current year)")
    parser.add_argument("--model", default=None,
                        help=f"Chat model used to phrase the questions "
                             f"(default: $LSB_GENERATOR_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL (default: $OPENAI_BASE_URL)")
    parser.add_argument("--api-key", default=None,
                        help="API key (default: $OPENAI_API_KEY, then .env)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
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
                        help="Keep every predicate that is not on the exclusion list, "
                             "instead of only the 198 allow-listed relation labels")
    parser.add_argument("--skip-verification", action="store_true",
                        help="Do not contact WDQS. Instances are written with "
                             "verified_with_sparql=false; use only for smoke tests")
    parser.add_argument("--dry-run", action="store_true",
                        help="Replace the model call with a deterministic template stub "
                             "(no API key required)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.num_questions <= 0:
        raise SystemExit("--num-questions must be positive")
    if args.candidate_pool < args.num_questions:
        logger.warning("--candidate-pool (%d) is below --num-questions (%d); "
                       "the run will stop short of the target",
                       args.candidate_pool, args.num_questions)

    input_path = resolve_input(args.input)
    started = datetime.now(timezone.utc)
    year = args.year or str(started.year)
    model = config.get("LSB_GENERATOR_MODEL", override=args.model, default=DEFAULT_MODEL)
    rng = random.Random(args.seed)
    random.seed(args.seed)  # the module rng is still used by third-party helpers
    stats = filters.FilterStats()

    # Fail before any long work if the destination is unwritable.
    provisional_output = Path(args.output) if args.output else default_output_path(0, year)
    ensure_parent(provisional_output)

    try:
        writer = QuestionWriter(model=model, base_url=args.base_url, api_key=args.api_key,
                                dry_run=args.dry_run, temperature=args.temperature,
                                max_tokens=args.max_tokens)
    except MissingCredential as exc:
        raise SystemExit(f"{exc}\n  Or pass --dry-run to exercise the pipeline without a model.")
    client = None if args.skip_verification else SparqlClient(endpoint=args.endpoint,
                                                              min_interval=args.min_interval)
    try:
        frame = load_triples(input_path, max_rows=args.max_rows, rng=rng, stats=stats)
        frame = filter_relations(frame, use_allowlist=not args.no_relation_allowlist, stats=stats)
        candidates = prepare_candidates(frame, pool_size=args.candidate_pool, rng=rng, stats=stats)
        verified = verify_candidates(candidates, client, target=args.num_questions, stats=stats,
                                     max_consecutive_errors=args.max_sparql_errors)
        qa_pairs = generate_questions(verified, writer, stats=stats)
    finally:
        writer.close()

    finished = datetime.now(timezone.utc)
    output_path = Path(args.output) if args.output else default_output_path(len(qa_pairs), year)
    ensure_parent(output_path)

    metadata = {
        "description": "Level 1 single-hop questions with SPARQL-verified unique answers",
        "level": 1,
        "year": year,
        "total_questions": len(qa_pairs),
        "sparql_verified_count": sum(1 for qa in qa_pairs if qa["verified_with_sparql"]),
        "verification_method": ("skipped (--skip-verification)" if args.skip_verification
                                else "SPARQL COUNT == 1 on the live endpoint"),
        "sparql_endpoint": None if client is None else client.endpoint,
        "model": writer.model_label,
        "requested_model": model,
        "question_source": writer.source_label,
        "dry_run": bool(args.dry_run),
        "seed": args.seed,
        "candidate_pool_requested": args.candidate_pool,
        "candidate_pool_actual": len(candidates),
        "num_questions_requested": args.num_questions,
        "relation_allowlist": not args.no_relation_allowlist,
        "input_file": str(input_path),
        "generator": "scripts/generate_level1.py",
        "generation_started_utc": started.isoformat(timespec="seconds"),
        "generation_finished_utc": finished.isoformat(timespec="seconds"),
    }
    output_path.write_text(
        json.dumps({"metadata": metadata, "qa_pairs": qa_pairs}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    report_path = Path(args.report) if args.report else output_path.with_suffix(".run_report.json")
    ensure_parent(report_path)
    report = {"metadata": metadata, "funnel": stats.to_dict(),
              "llm_calls": writer.calls, "llm_failures": writer.failures}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nLevel 1 generation complete: {len(qa_pairs)} questions")
    print(f"  questions : {output_path}")
    print(f"  run report: {report_path}")
    print()
    print(stats.render())
    if qa_pairs:
        print("\nExamples:")
        for qa in qa_pairs[:3]:
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer']}  [{qa['source_triple']['predicate_label']}, "
                  f"verified={qa['verified_with_sparql']}]")
    if len(qa_pairs) < args.num_questions:
        logger.warning("Produced %d of the %d requested questions; see the drop reasons above.",
                       len(qa_pairs), args.num_questions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
