#!/usr/bin/env python3
"""Oracle triple injection: measure reasoning with retrieval removed.

Every LiveSearchBench instance is generated from a small set of gold Wikidata
triples. This runner renders those triples back into short natural-language
passages, drops them straight into the prompt, and asks the question. No search
engine, no retriever, no entity linking: the evidence the question was built
from is simply handed to the model.

That isolates one half of the paper's dual failure mode. A model that still
fails here failed to *reason* over evidence it was given; a model that succeeds
here but fails under ``RAG.py`` failed to *retrieve*. Comparing this script's
accuracy against ``CoT.py`` (no evidence) and ``RAG.py`` (retrieved evidence)
is what separates the two.

Where the gold triples come from, per level:

============ ===============================================================
level 1      ``reasoning_chain`` / ``source_triple``
level 2      ``reasoning_chain`` / ``constraint_info.constraints``
level 3      ``reasoning_chain`` / ``abstraction_info.original_constraints``
any          the ``sparql_verification`` program, as a last resort
============ ===============================================================

The SPARQL fallback is what makes ``demo.json`` usable: those 30 items carry no
``reasoning_chain``, only a COUNT program. Entity and property IDs recovered
from it are opaque (``Q4810517``, ``P4552``) until ``--resolve-labels`` looks
their English labels up on Wikidata; labels are cached on disk so a resolved
split replays offline.

Examples:
    # offline smoke test, no API key needed
    python scripts/eval/oracle.py demo.json --dry-run --resolve-labels

    # real run
    python scripts/eval/oracle.py bench/2025/level2.json \
        --model gpt-4o-mini --concurrency 8

    # inject 5 unrelated triples per item as distractors
    python scripts/eval/oracle.py bench/2025/level1.json \
        --model gpt-4o-mini --distractors 5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from livesearchbench import config, dataio, scoring  # noqa: E402

logger = logging.getLogger("livesearchbench.eval.oracle")

METHOD = "oracle"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONCURRENCY = 4
RETRY_INITIAL_DELAY = 0.5
RETRY_MAX_DELAY = 10.0
MODEL_MAX_ATTEMPTS = 5
DEFAULT_LABEL_CACHE = REPO_ROOT / "outputs" / "cache" / "wikidata_labels.json"

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
#: "parent organization: California Military Department"
_CONSTRAINT_RE = re.compile(r"^\s*(?P<pred>[^:]+?)\s*:\s*(?P<obj>.+?)\s*$")
#: "abstract [shares border with: Ciudad Real] As [...]" -- only the original half is gold.
_ABSTRACT_RE = re.compile(r"abstract\s*\[\s*(?P<orig>[^\]]+?)\s*\]", re.IGNORECASE)
#: A basic triple pattern inside a verification program's WHERE clause.
_SPARQL_TRIPLE_RE = re.compile(
    r"(?P<s>wd:Q\d+|\?[A-Za-z_]\w*)\s+"
    r"(?P<p>wdt:P\d+|p:P\d+|ps:P\d+)\s+"
    r"(?P<o>wd:Q\d+|\?[A-Za-z_]\w*|\"[^\"]*\")"
)
_ID_RE = re.compile(r"\b(?:wd:|wdt:|p:|ps:)?([QP]\d+)\b")


class Triple(NamedTuple):
    """One gold fact, already resolved to human-readable labels where possible."""

    subject: str
    predicate: str
    obj: str
    source: str


# ---------------------------------------------------------------------------
# Gold-triple extraction
# ---------------------------------------------------------------------------

def clean_text(value) -> str:
    return " ".join(str(value or "").split())


def _split_constraint(text: str, answer: str, source: str) -> Optional[Triple]:
    """Turn a ``"predicate: object"`` constraint string into a triple.

    The subject of a level-2/level-3 constraint is the answer entity itself:
    the question asks which entity satisfies all of them.
    """
    match = _CONSTRAINT_RE.match(str(text or ""))
    if not match:
        return None
    predicate = clean_text(match.group("pred"))
    obj = clean_text(match.group("obj"))
    if not predicate or not obj:
        return None
    return Triple(clean_text(answer) or "the answer entity", predicate, obj, source)


def wikidata_ids(sparql: str) -> List[str]:
    """Every Q/P identifier mentioned by a verification program, in order."""
    seen: Set[str] = set()
    out: List[str] = []
    for ident in _ID_RE.findall(sparql or ""):
        if ident not in seen:
            seen.add(ident)
            out.append(ident)
    return out


def triples_from_sparql(sparql: str, answer: str, labels: Optional[Dict[str, str]] = None) -> List[Triple]:
    """Recover triples from a COUNT verification program.

    Released programs contain exactly one variable, which stands for the gold
    answer (``?object`` in a level-1 program, ``?entity`` in a level-2 one), so
    every variable is substituted with the answer string. Bare identifiers are
    replaced by their English label when ``labels`` supplies one.
    """
    labels = labels or {}
    answer_text = clean_text(answer) or "the answer entity"

    def render(token: str) -> str:
        token = token.strip()
        if token.startswith("?"):
            return answer_text
        if token.startswith('"'):
            return token.strip('"')
        ident = token.split(":", 1)[-1]
        return labels.get(ident, ident)

    out: List[Triple] = []
    for match in _SPARQL_TRIPLE_RE.finditer(sparql or ""):
        out.append(Triple(render(match.group("s")), render(match.group("p")),
                          render(match.group("o")), "sparql_verification"))
    return out


def extract_gold_triples(item: Dict, *, labels: Optional[Dict[str, str]] = None) -> List[Triple]:
    """Collect the gold provenance triples for one instance.

    Structured provenance is preferred and the SPARQL program is consulted only
    when none is present, so that ``bench/`` items never depend on parsing.
    Duplicates are removed case-insensitively while preserving order.
    """
    answer = item.get("answer", "")
    found: List[Triple] = []

    for row in item.get("reasoning_chain") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            found.append(Triple(clean_text(row[0]), clean_text(row[1]), clean_text(row[2]), "reasoning_chain"))

    source_triple = item.get("source_triple")
    if isinstance(source_triple, dict):
        subject = source_triple.get("subject_label") or source_triple.get("subject_id")
        predicate = source_triple.get("predicate_label") or source_triple.get("predicate_id")
        obj = source_triple.get("object_label") or source_triple.get("object_id")
        if subject and predicate and obj:
            found.append(Triple(clean_text(subject), clean_text(predicate), clean_text(obj), "source_triple"))

    constraint_info = item.get("constraint_info")
    if isinstance(constraint_info, dict):
        for text in constraint_info.get("constraints") or []:
            triple = _split_constraint(text, answer, "constraint_info")
            if triple:
                found.append(triple)

    abstraction_info = item.get("abstraction_info")
    if isinstance(abstraction_info, dict):
        for text in abstraction_info.get("original_constraints") or []:
            triple = _split_constraint(text, answer, "abstraction_info")
            if triple:
                found.append(triple)
        # The abstracted strings embed the original constraint in brackets.
        for text in abstraction_info.get("abstract_constraints") or []:
            match = _ABSTRACT_RE.search(str(text))
            if match:
                triple = _split_constraint(match.group("orig"), answer, "abstraction_info")
                if triple:
                    found.append(triple)

    if not found:
        found.extend(triples_from_sparql(item.get("sparql_verification", ""), answer, labels))

    deduped: List[Triple] = []
    seen: Set[Tuple[str, str, str]] = set()
    for triple in found:
        key = (triple.subject.lower(), triple.predicate.lower(), triple.obj.lower())
        if key in seen or not (triple.subject and triple.predicate and triple.obj):
            continue
        seen.add(key)
        deduped.append(triple)
    return deduped


# ---------------------------------------------------------------------------
# Label resolution
# ---------------------------------------------------------------------------

def load_label_cache(path) -> Dict[str, str]:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable label cache %s: %s", path, exc)
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def resolve_labels(ids: Sequence[str], *, cache_path, lang: str = "en",
                   batch_size: int = 50) -> Dict[str, str]:
    """Look English labels up for Q/P identifiers, memoised on disk.

    Network access happens only for identifiers missing from the cache, so a
    split resolved once replays with no connectivity at all.
    """
    labels = load_label_cache(cache_path)
    missing = [i for i in dict.fromkeys(ids) if i not in labels]
    if not missing:
        return labels

    from livesearchbench.http import PoliteSession, RequestFailed

    logger.info("Resolving %d Wikidata label(s) (%d already cached)", len(missing), len(labels))
    with PoliteSession(component="LiveSearchBench-Oracle", min_interval=0.2) as session:
        for start in range(0, len(missing), batch_size):
            chunk = missing[start:start + batch_size]
            try:
                data = session.wikidata_api({
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "labels",
                    "languages": lang,
                })
            except RequestFailed as exc:
                raise RuntimeError(
                    f"Could not resolve Wikidata labels: {exc}\n"
                    f"  Re-run without --resolve-labels to keep bare Q/P identifiers."
                ) from exc
            for ident, entity in (data.get("entities") or {}).items():
                label = ((entity.get("labels") or {}).get(lang) or {}).get("value")
                if label:
                    labels[ident] = label

    unresolved = [i for i in missing if i not in labels]
    if unresolved:
        logger.warning("%d identifier(s) have no %s label and stay as bare ids (first: %s); "
                       "these are usually redirected or deleted entities",
                       len(unresolved), lang, ", ".join(unresolved[:5]))

    out_path = dataio.ensure_parent(cache_path)
    out_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2, sort_keys=True),
                        encoding="utf-8")
    logger.info("Label cache now holds %d entries (%s)", len(labels), out_path)
    return labels


_BARE_ID_RE = re.compile(r"^[QP]\d+$")


def warn_if_unresolved(triples_per_item: Sequence[Sequence[Triple]], *, resolved: bool) -> float:
    """Warn when many triples still carry bare Q/P ids instead of labels.

    Opaque ids make an oracle context unreadable and make a BM25 corpus
    unmatchable, so a run without ``--resolve-labels`` on a split that needs it
    would otherwise look like a genuine model or retrieval failure.
    """
    total = sum(len(group) for group in triples_per_item)
    if not total:
        return 0.0
    bare = sum(1 for group in triples_per_item for t in group
               if _BARE_ID_RE.match(t.subject) or _BARE_ID_RE.match(t.obj)
               or _BARE_ID_RE.match(t.predicate))
    share = bare / total
    if share > 0.05:
        logger.warning(
            "%.0f%% of gold triples (%d/%d) still carry bare Wikidata ids such as Q42/P31.%s",
            100 * share, bare, total,
            "" if resolved else " Pass --resolve-labels to fetch their English labels.")
    return share


# ---------------------------------------------------------------------------
# Verbalisation and prompting
# ---------------------------------------------------------------------------

def verbalize_triple(triple: Triple) -> str:
    """Render one triple as a passage line.

    The schema-faithful ``subject | predicate | object`` form is used rather
    than a generated sentence: Wikidata predicate labels ("shares border with",
    "country of origin") do not slot into a fixed template without producing
    ungrammatical text that would itself confound the diagnostic.
    """
    return f"{triple.subject} | {triple.predicate} | {triple.obj}"


def build_oracle_context(triples: Sequence[Triple]) -> str:
    """Number the verbalised triples so the model can cite them."""
    if not triples:
        return "(no supporting facts are available for this question)"
    return "\n".join(f"{i}. {verbalize_triple(t)}" for i, t in enumerate(triples, 1))


def build_prompt(question: str, context: str) -> str:
    if not question.endswith("?"):
        question += "?"
    return (
        "You are given verified facts extracted from Wikidata, one per line, in the form "
        "subject | property | value.\n\n"
        "Facts:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using these facts. Think step by step, then provide ONLY the final "
        "answer inside <answer> and </answer> tags. Do not output anything else after the tags. "
        "For example, <answer> Beijing </answer>."
    )


def extract_answer(content: str) -> str:
    matches = _ANSWER_RE.findall(content or "")
    return matches[-1].strip() if matches else str(content or "").strip()


def call_model(client, messages, *, model_name: str, max_tokens: int, temperature: float,
               max_attempts: int = MODEL_MAX_ATTEMPTS) -> str:
    """Bounded exponential-backoff wrapper around the chat completion call."""
    delay = RETRY_INITIAL_DELAY
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - the SDK raises many types
            last_error = exc
            logger.warning("Model call failed (attempt %d/%d): %s", attempt, max_attempts, exc)
            if attempt < max_attempts:
                time.sleep(delay)
                delay = min(delay * 2, RETRY_MAX_DELAY)
    raise RuntimeError(f"Model call failed after {max_attempts} attempts: {last_error}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def sample_distractors(all_triples: Sequence[Sequence[Triple]], index: int, n: int,
                       rng: random.Random) -> List[Triple]:
    """Draw ``n`` triples belonging to other instances of the same split."""
    pool = [t for i, group in enumerate(all_triples) if i != index for t in group]
    if not pool or n <= 0:
        return []
    return rng.sample(pool, min(n, len(pool)))


def evaluate_item(
    item: Dict,
    triples: Sequence[Triple],
    *,
    client,
    model_name: str,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
) -> Dict:
    question = clean_text(item.get("question"))
    expected = item.get("answer", "")
    aliases = item.get("answer_aliases") or []
    context = build_oracle_context(triples)
    prompt = build_prompt(question, context)

    context_has_answer = scoring.contains_match(context, expected, aliases)
    record: Dict = {
        "question": question,
        "expected_answer": expected,
        # Carried through so a later re-score by scripts/analysis reproduces
        # exactly the numbers written here.
        "answer_aliases": list(aliases),
        "level": item.get("level"),
        "n_oracle_triples": len(triples),
        "oracle_triple_sources": sorted({t.source for t in triples}),
        "oracle_context_has_answer": context_has_answer,
        "oracle_context": context,
    }

    if dry_run:
        record.update({
            "model_answer": "",
            "reasoning_process": "",
            "is_correct": False,
            "dry_run": True,
            "error": None,
        })
        record.update({k: 0.0 for k in ("exact_match", "token_f1", "contains_match")})
        return record

    try:
        content = call_model(
            client, [{"role": "user", "content": prompt}], model_name=model_name,
            max_tokens=max_tokens, temperature=temperature,
        )
        error = None
    except RuntimeError as exc:
        logger.error("Giving up on question %r: %s", question[:60], exc)
        content = ""
        error = str(exc)

    answer = extract_answer(content)
    scores = scoring.score_item(answer, expected, aliases)
    record.update({
        "model_answer": answer,
        "reasoning_process": content,
        # Normalised exact match, matching DA/CoT/RAG. Previously this was
        # contains_match, so accuracy from this runner was not comparable
        # with theirs. contains_match is still recorded alongside.
        "is_correct": bool(scores["exact_match"]),
        "dry_run": False,
        "error": error,
    })
    record.update(scores)
    return record


def build_summary(results: Sequence[Dict], *, model_name: str, dry_run: bool,
                  distractors: int) -> Dict:
    total = len(results)
    scored = [r for r in results if not r.get("dry_run")]
    correct = sum(1 for r in scored if r.get("is_correct"))
    metrics = scoring.aggregate(scored) if scored else {}
    with_answer = sum(1 for r in results if r.get("oracle_context_has_answer"))
    no_triples = sum(1 for r in results if not r.get("n_oracle_triples"))
    return {
        "method": METHOD,
        "model": model_name,
        "dry_run": dry_run,
        "distractors_per_item": distractors,
        "total_questions": total,
        "scored_questions": len(scored),
        "correct_answers": correct,
        "accuracy": (correct / len(scored)) if scored else None,
        "errors": sum(1 for r in results if r.get("error")),
        # Coverage is the ceiling this diagnostic can reach: it is the share of
        # items whose injected context literally contains the gold answer.
        "oracle_context_coverage": (with_answer / total) if total else 0.0,
        "items_without_triples": no_triples,
        "mean_triples_per_item": (sum(r.get("n_oracle_triples", 0) for r in results) / total)
                                 if total else 0.0,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oracle.py",
        description="Oracle triple injection: put an instance's gold Wikidata triples straight "
                    "into the prompt, bypassing retrieval, to measure reasoning in isolation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Compare the accuracy reported here against CoT.py (no evidence) and RAG.py "
               "(retrieved evidence) to tell a reasoning failure from a retrieval failure.",
    )
    parser.add_argument("data_positional", nargs="?", metavar="DATA",
                        help="benchmark split, e.g. bench/2025/level1.json or demo.json")
    parser.add_argument("--data", dest="data_flag", metavar="PATH",
                        help="same as the positional DATA argument")
    parser.add_argument("--model", default="", help="model name passed to the chat completions API")
    parser.add_argument("--concurrency", "--threads", dest="concurrency", type=int,
                        default=DEFAULT_CONCURRENCY, help="parallel model calls (default: 4)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="maximum tokens to generate per answer (default: 2048)")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N instances (0 = all)")
    parser.add_argument("--distractors", type=int, default=0,
                        help="inject N gold triples borrowed from other instances of the same "
                             "split, shuffled in, as an oracle-with-noise ablation (default: 0)")
    parser.add_argument("--seed", type=int, default=0, help="seed for distractor sampling")
    parser.add_argument("--resolve-labels", action="store_true",
                        help="look up English labels for Q/P ids recovered from SPARQL "
                             "programs (needed for demo.json; results are cached on disk)")
    parser.add_argument("--label-cache", default=str(DEFAULT_LABEL_CACHE),
                        help=f"label cache file (default: {DEFAULT_LABEL_CACHE})")
    parser.add_argument("--output-dir", default=None,
                        help="where to write results (default: outputs/evaluations/<year>)")
    parser.add_argument("--resume", nargs="?", const="", default=None, metavar="RESULTS_JSON",
                        help="skip questions already answered. With no value, reuses the newest "
                             "matching results file in the output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and score the oracle contexts but never call the model; "
                             "needs no API key and still reports context coverage")
    parser.add_argument("--show-context", type=int, default=0, metavar="N",
                        help="print the fully rendered prompt for the first N instances")
    parser.add_argument("--base-url", default=None, help="override OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=None, help="override OPENAI_API_KEY")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    args.data = args.data_flag or args.data_positional
    if not args.data:
        parser.error("a dataset path is required (positional DATA or --data)")
    if not args.dry_run and not args.model:
        parser.error("--model is required unless --dry-run is given")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    return args


def find_resume_file(output_dir: Path, model_name: str, level: str,
                     method: str = METHOD) -> Optional[Path]:
    """Newest results file in ``output_dir`` written by this method and model.

    ``method`` is a parameter because wiki_corpus.py reuses this helper and
    must not pick up oracle.py's result files.
    """
    safe_model = re.sub(r"[^A-Za-z0-9]+", "_", str(model_name)).strip("_") or "model"
    pattern = f"{level}_{method}_{safe_model}*_results.json"
    candidates = sorted(output_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_resume(path: Path) -> Dict[str, Dict]:
    """Map question text -> record, for records that carry a real answer."""
    records, _ = dataio.load_results(path)
    done = {r["question"]: r for r in records
            if r.get("question") and r.get("model_answer") and not r.get("dry_run")}
    logger.info("Resuming from %s: %d completed question(s)", path, len(done))
    return done


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    raw_items, meta = dataio.load_instances(args.data)
    items = [dataio.normalize_instance(it) for it in raw_items]
    if args.limit:
        items = items[:args.limit]
    logger.info("Loaded %d instance(s) from %s (level=%s year=%s)",
                len(items), args.data, meta.get("level"), meta.get("year"))

    # Resolve labels before any long work so a network problem surfaces now.
    labels: Dict[str, str] = {}
    if args.resolve_labels:
        needed: List[str] = []
        for item in items:
            provisional = extract_gold_triples(item)
            if any(t.source == "sparql_verification" for t in provisional):
                needed.extend(wikidata_ids(item.get("sparql_verification", "")))
        labels = resolve_labels(needed, cache_path=args.label_cache) if needed else {}
        if not needed:
            logger.info("--resolve-labels: no instance needs the SPARQL fallback, nothing to resolve")
    else:
        labels = load_label_cache(args.label_cache)

    triples_per_item = [extract_gold_triples(item, labels=labels) for item in items]
    warn_if_unresolved(triples_per_item, resolved=args.resolve_labels)
    if args.distractors:
        rng = random.Random(args.seed)
        merged = []
        for idx, gold in enumerate(triples_per_item):
            noisy = list(gold) + sample_distractors(triples_per_item, idx, args.distractors, rng)
            rng.shuffle(noisy)
            merged.append(noisy)
        triples_per_item = merged

    empty = sum(1 for t in triples_per_item if not t)
    if empty:
        logger.warning("%d/%d instance(s) yielded no gold triples; their oracle context is empty",
                       empty, len(items))

    for idx in range(min(args.show_context, len(items))):
        print(f"\n===== oracle prompt {idx + 1}/{len(items)} =====")
        print(build_prompt(clean_text(items[idx]["question"]),
                           build_oracle_context(triples_per_item[idx])))
        print(f"----- gold answer: {items[idx]['answer']}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        REPO_ROOT / "outputs" / "evaluations" / str(meta.get("year") or "unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)

    done: Dict[str, Dict] = {}
    if args.resume is not None:
        resume_path = Path(args.resume) if args.resume else find_resume_file(
            output_dir, args.model, str(meta.get("level") or "unknown"))
        if resume_path and Path(resume_path).is_file():
            done = load_resume(Path(resume_path))
        else:
            logger.warning("--resume given but no prior results file was found in %s", output_dir)

    client = None
    if not args.dry_run:
        base_url, api_key = config.openai_credentials(base_url=args.base_url, api_key=args.api_key)
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info("Calling model %r at %s with concurrency %d",
                    args.model, base_url, args.concurrency)
    else:
        logger.warning("DRY RUN: no model will be called; every model_answer will be empty "
                       "and accuracy is reported as null")

    counter = {"n": 0}
    counter_lock = threading.Lock()

    def run_one(payload: Tuple[int, Dict]) -> Dict:
        idx, item = payload
        cached = done.get(clean_text(item.get("question")))
        if cached is not None:
            return cached
        record = evaluate_item(
            item, triples_per_item[idx], client=client, model_name=args.model,
            max_tokens=args.max_tokens, temperature=args.temperature, dry_run=args.dry_run,
        )
        with counter_lock:
            counter["n"] += 1
            logger.info("[%d/%d] %s -> %r (gold %r)", counter["n"], len(items),
                        "dry-run" if args.dry_run else ("correct" if record["is_correct"] else "wrong"),
                        record["model_answer"][:60], record["expected_answer"])
        return record

    payloads = list(enumerate(items))
    if args.dry_run or args.concurrency == 1:
        results = [run_one(p) for p in payloads]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(run_one, payloads))

    summary = build_summary(results, model_name=args.model or "(dry-run)",
                            dry_run=args.dry_run, distractors=args.distractors)
    # dataio.save_run merges this metadata OVER the summary, and every released
    # split's header already defines total_questions, so the split's own header
    # is nested rather than spread to keep the run's counts intact.
    run_meta = {"level": meta.get("level"), "year": meta.get("year"),
                "dataset_metadata": meta}
    paths = dataio.save_run(
        results=results, summary=summary, method=METHOD,
        model_name=args.model or "dry_run", data_path=args.data,
        output_dir=str(output_dir), metadata=run_meta,
    )

    print("\n=== oracle triple injection ===")
    print(f"instances                : {summary['total_questions']}")
    print(f"mean gold triples / item : {summary['mean_triples_per_item']:.2f}")
    print(f"items without triples    : {summary['items_without_triples']}")
    print(f"context contains answer  : {summary['oracle_context_coverage']:.1%}")
    if summary["accuracy"] is None:
        print("accuracy                 : n/a (dry run, no model was called)")
    else:
        print(f"accuracy (contains)      : {summary['accuracy']:.2%} "
              f"({summary['correct_answers']}/{summary['scored_questions']})")
        overall = summary["metrics"].get("overall", {})
        for metric in ("exact_match", "token_f1", "contains_match"):
            cell = overall.get(metric, {})
            print(f"{metric:<25}: {cell.get('value')} "
                  f"[{cell.get('ci_low')}, {cell.get('ci_high')}]")
    print(f"results                  : {paths['results']}")
    print(f"summary                  : {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
