#!/usr/bin/env python3
"""Direct-Answer (DA) baseline runner: no search, no chain of thought.

The model sees the question and is asked for the final answer only, wrapped in
``<answer> </answer>``. This is the closed-book lower bound in the paper.

Changes relative to the first release:

* Credentials resolve through :mod:`livesearchbench.config` (``--api-key`` /
  ``--base-url`` > ``$OPENAI_API_KEY`` / ``$OPENAI_BASE_URL`` > ``.env``)
  instead of the ``"YOUR_API_KEY"`` module-level placeholders.
* Datasets load through :func:`livesearchbench.dataio.load_instances`, so the
  bare-list ``demo.json`` works as well as the ``{"qa_pairs": ...}`` splits.
  The old ``test_data["qa_pairs"]`` crashed on ``demo.json``.
* Scoring uses :func:`livesearchbench.scoring.score_item`; every record now
  carries ``exact_match``, ``token_f1`` and ``contains_match``. ``is_correct``
  is kept for backward compatibility but is now **normalised exact match**,
  where it used to be the much more permissive substring containment (now
  reported as ``contains_match``). Numbers from this runner are therefore not
  directly comparable with numbers printed by the first release.
* Records stream to a JSONL sidecar as they are produced, so an interrupted run
  keeps its work and can be continued with ``--resume``.
* ``--n-samples N`` draws N generations per question and records all of them,
  so ``scripts/analysis/score.py`` can compute pass@k.
* ``--dry-run`` swaps in a local stub model, so the whole pipeline can be
  exercised end to end without any credentials.
* Model-call retries are bounded (the old loop retried forever).

Examples:
    python scripts/eval/DA.py demo.json --model gpt-4o-mini
    python scripts/eval/DA.py bench/2025/level1.json --model gpt-4o-mini --n-samples 5 --resume
    python scripts/eval/DA.py demo.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from livesearchbench import config, dataio, scoring  # noqa: E402

LOGGER = logging.getLogger("livesearchbench.eval.DA")

METHOD = "DA"
PROMPT_TEMPLATE = (
    "Answer the given question. Provide ONLY the final result inside <answer> and </answer>. "
    "Do not output anything else. For example, <answer> Beijing </answer>. Question: {question}"
)


class AbortRun(RuntimeError):
    """Raised inside a worker once the run has been declared unrecoverable."""


class FirstFailureGuard:
    """Abort the whole run when the very first call of a kind fails.

    A failure on the first call is almost always a configuration problem (wrong
    key, wrong base URL, wrong model name) rather than a transient one, so it is
    far better to stop immediately than to burn the entire split and report a
    silently depressed score.
    """

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.abort = threading.Event()
        self.reason = ""
        self._lock = threading.Lock()
        self._decided = False

    def record(self, ok: bool, detail: str = "") -> None:
        with self._lock:
            if self._decided:
                return
            self._decided = True
            if not ok:
                self.reason = detail
                self.abort.set()

    def check(self) -> None:
        if self.abort.is_set():
            raise AbortRun(f"first {self.kind} call failed: {self.reason}")


class StubModel:
    """Offline stand-in for the chat API, used by ``--dry-run``.

    Answers are synthetic: a deterministic, seeded fraction of them echo the
    gold answer and the rest return a fixed canned string. Scores produced in
    dry-run mode are meaningless and every record is tagged ``"dry_run": true``.
    """

    name = "dry-run-stub"

    def __init__(self, *, correct_rate: float = 0.5, canned: str = "Beijing") -> None:
        self.correct_rate = correct_rate
        self.canned = canned

    def complete(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        gold: str = "",
        index: int = 0,
        sample_index: int = 0,
        **_: Any,
    ) -> str:
        rng = random.Random(f"{index}:{sample_index}")
        answer = str(gold) if rng.random() < self.correct_rate else self.canned
        return f"<answer> {answer} </answer>"


class ChatModel:
    """OpenAI-compatible chat client with bounded retries."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        max_retries: int = 5,
        initial_delay: float = 0.5,
        max_delay: float = 10.0,
    ) -> None:
        from openai import OpenAI  # imported lazily so --dry-run needs no SDK

        self.name = name
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.max_retries = max(1, int(max_retries))
        self.initial_delay = initial_delay
        self.max_delay = max_delay

    def complete(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **_: Any,
    ) -> str:
        delay = self.initial_delay
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.name,
                    messages=list(messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001 - the SDK raises many types
                last_error = f"{type(exc).__name__}: {exc}"
                LOGGER.warning("Model call failed (attempt %d/%d): %s",
                               attempt, self.max_retries, last_error)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, self.max_delay)
                continue
            content = response.choices[0].message.content
            if content is None:
                last_error = "the endpoint returned a message with no content"
                LOGGER.warning("Model call returned empty content (attempt %d/%d)",
                               attempt, self.max_retries)
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay = min(delay * 2, self.max_delay)
                continue
            return content
        raise RuntimeError(
            f"Model '{self.name}' failed after {self.max_retries} attempts. Last error: {last_error}"
        )


class JsonlWriter:
    """Append-only, thread-safe, flushed-per-record sidecar writer."""

    def __init__(self, path: Path, *, append: bool) -> None:
        self.path = dataio.ensure_parent(Path(path))
        self._lock = threading.Lock()
        self._handle = open(self.path, "a" if append else "w", encoding="utf-8")

    def write(self, record: Dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def extract_answer(content: str) -> str:
    """Pull the ``<answer>`` payload out of a completion, else return it whole."""
    match = re.search(r"<answer>(.*?)</answer>", content or "", re.DOTALL)
    if match:
        return match.group(1).strip()
    return (content or "").strip()


def record_key(index: Any, question: str) -> str:
    return f"{index}::{question}"


def safe_model_tag(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(model_name)).strip("_") or "model"


def default_partial_path(
    *,
    output_dir: Optional[str],
    meta: Dict[str, Any],
    model_name: str,
    budget: Optional[int] = None,
) -> Path:
    """Deterministic sidecar path, so ``--resume`` can find a previous run."""
    level = meta.get("level") or "unknown"
    year = meta.get("year") or "unknown"
    tag = f"_maxiter_{budget}" if budget is not None else ""
    directory = Path(output_dir or os.path.join("outputs", "evaluations", str(year)))
    return directory / f"{level}_{METHOD}_{safe_model_tag(model_name)}{tag}_partial.jsonl"


def load_partial(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read a sidecar written by an earlier run; later lines win."""
    records: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return records
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            record = json.loads(raw)
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring malformed line %d of %s", lineno, path)
            continue
        records[record_key(record.get("index"), record.get("question", ""))] = record
    return records


def build_record(
    *,
    index: int,
    question: str,
    item: Dict[str, Any],
    samples: List[Dict[str, Any]],
    dry_run: bool,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble one per-question record from its samples."""
    primary = samples[0]
    record = {
        "index": index,
        "question": question,
        "expected_answer": item.get("answer", ""),
        "answer_aliases": item.get("answer_aliases") or [],
        "level": item.get("level"),
        "model_answer": primary["model_answer"],
        "is_correct": bool(primary["exact_match"]),
        "exact_match": primary["exact_match"],
        "token_f1": primary["token_f1"],
        "contains_match": primary["contains_match"],
        "n_samples": len(samples),
        "n_correct": int(sum(s["exact_match"] for s in samples)),
        "samples": samples,
        "reasoning_process": primary.get("reasoning_process", []),
    }
    if dry_run:
        record["dry_run"] = True
    if error:
        record["error"] = error
    return record


def process_item(
    index: int,
    item: Dict[str, Any],
    *,
    model: Any,
    guard: FirstFailureGuard,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    total: int,
    dry_run: bool,
) -> Dict[str, Any]:
    guard.check()
    question = str(item.get("question", "")).strip()
    if not question.endswith("?"):
        question += "?"
    gold = item.get("answer", "")
    aliases = item.get("answer_aliases") or []
    prompt = PROMPT_TEMPLATE.format(question=question)

    samples: List[Dict[str, Any]] = []
    error: Optional[str] = None
    for sample_index in range(n_samples):
        guard.check()
        try:
            content = model.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                gold=gold,
                index=index,
                sample_index=sample_index,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            guard.record(False, str(exc))
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Item %d sample %d failed: %s", index, sample_index, error)
            content = ""
        else:
            guard.record(True)
        answer = extract_answer(content)
        scores = scoring.score_item(answer, gold, aliases)
        sample = {
            "sample_index": sample_index,
            "model_answer": answer,
            "reasoning_process": [  # holds the untruncated completion
                {"type": "prompt", "content": prompt},
                {"type": "model_output", "step": 0, "content": content},
            ],
            **scores,
        }
        if error:
            sample["error"] = error
        samples.append(sample)
        if error:
            # The endpoint is broken; drawing the remaining samples cannot help.
            break

    record = build_record(index=index, question=question, item=item,
                          samples=samples, dry_run=dry_run, error=error)
    LOGGER.info("[%d/%d] EM=%d F1=%.2f | gold=%r | pred=%r",
                index + 1, total, int(record["exact_match"]),
                record["token_f1"], gold, record["model_answer"])
    return record


def flatten_samples(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand per-question records into one flat record per sample.

    ``scripts/analysis/score.py`` groups samples by question text, so pass@k
    needs the samples as sibling records rather than nested. With the default
    ``--n-samples 1`` this is one record per question, exactly as before.
    """
    flat: List[Dict[str, Any]] = []
    for record in records:
        for sample in record.get("samples") or []:
            row: Dict[str, Any] = {
                "index": record.get("index"),
                "sample_index": sample.get("sample_index", 0),
                "question": record.get("question", ""),
                "expected_answer": record.get("expected_answer", ""),
                "model_answer": sample.get("model_answer", ""),
                "is_correct": bool(sample.get("exact_match")),
                "exact_match": sample.get("exact_match", 0.0),
                "token_f1": sample.get("token_f1", 0.0),
                "contains_match": sample.get("contains_match", 0.0),
                "answer_aliases": record.get("answer_aliases") or [],
                "level": record.get("level"),
                "n_samples": record.get("n_samples", 1),
                "n_correct": record.get("n_correct", 0),
                "reasoning_process": sample.get("reasoning_process", []),
            }
            if "error" in sample:
                row["error"] = sample["error"]
            if record.get("dry_run"):
                row["dry_run"] = True
            flat.append(row)
    return flat


def summarize(
    *,
    records: Sequence[Dict[str, Any]],
    flat: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    model_name: str,
    elapsed: float,
    partial_path: Path,
) -> Dict[str, Any]:
    """Summarise a run.

    Metrics are computed over every sample, so with ``--n-samples > 1`` they are
    sample-averaged rather than per-question; pass@k is reported separately.
    """
    total = len(records)
    correct = sum(1 for r in flat if r.get("is_correct"))
    metrics = scoring.aggregate(flat, group_key="level")
    summary: Dict[str, Any] = {
        "method": METHOD,
        "model": model_name,
        "search_enabled": False,
        "total_questions": total,
        "total_samples": len(flat),
        "correct_answers": correct,
        "accuracy": round(correct / len(flat), 4) if flat else 0.0,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "metrics": metrics,
        "items_with_errors": sum(1 for r in records if r.get("error")),
        "elapsed_seconds": round(elapsed, 1),
        "test_file": args.data,
        "partial_file": str(partial_path),
        "dry_run": bool(args.dry_run),
    }
    if args.n_samples > 1:
        summary["pass_at_k"] = {
            f"pass@{k}": round(
                100.0 * sum(
                    scoring.pass_at_k(r.get("n_samples", 1), r.get("n_correct", 0), k)
                    for r in records
                ) / total, 2)
            for k in range(1, args.n_samples + 1)
        } if total else {}
    return summary


def print_summary(summary: Dict[str, Any], paths: Dict[str, str]) -> None:
    overall = summary["metrics"]["overall"]
    print(f"\n=== {summary['method']} results: {summary['model']} ===")
    print(f"Questions        : {summary['total_questions']}"
          + (f" ({summary['total_samples']} samples)"
             if summary["total_samples"] != summary["total_questions"] else ""))
    for metric in ("exact_match", "token_f1", "contains_match"):
        stat = overall[metric]
        print(f"{metric:<17}: {stat['value']:6.2f}  (95% CI {stat['ci_low']:.2f}-{stat['ci_high']:.2f})")
    by_level = summary["metrics"].get("by_level") or {}
    for level, stat in by_level.items():
        print(f"  level {level:<10}: EM {stat['exact_match']['value']:6.2f}  "
              f"F1 {stat['token_f1']['value']:6.2f}  n={stat['n']}")
    if summary.get("pass_at_k"):
        print("pass@k           : " + ", ".join(f"{k}={v:.2f}" for k, v in summary["pass_at_k"].items()))
    if summary.get("items_with_errors"):
        print(f"Items with errors: {summary['items_with_errors']}")
    if summary.get("dry_run"):
        print("NOTE: --dry-run was used. These numbers come from a stub model and mean nothing.")
    print(f"Results          : {paths['results']}")
    print(f"Summary          : {paths['summary']}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="DA.py",
        description="Direct-answer (closed-book, no search) baseline for LiveSearchBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="Benchmark split, e.g. demo.json or bench/2025/level1.json")
    parser.add_argument("--model", default=None,
                        help="Model name sent to the OpenAI-compatible endpoint "
                             "(required unless --dry-run)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL; falls back to $OPENAI_BASE_URL, then .env, "
                             "then https://api.openai.com/v1")
    parser.add_argument("--api-key", default=None,
                        help="API key; falls back to $OPENAI_API_KEY, then .env")
    parser.add_argument("--threads", type=int, default=4, help="Questions answered in parallel")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Generation cap per call")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Generations per question; >1 records every sample so that "
                             "scripts/analysis/score.py can compute pass@k")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Bounded retries per model call before the item is marked failed")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--output-dir", default=None,
                        help="Where results land; default outputs/evaluations/<year>")
    parser.add_argument("--partial-file", default=None,
                        help="JSONL sidecar written as the run proceeds; default is derived from "
                             "the split and model name so --resume can find it")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse records already present in the sidecar and only run what is missing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use an offline stub model instead of the API; needs no credentials")
    parser.add_argument("--dry-run-correct-rate", type=float, default=0.5,
                        help="Fraction of stub answers that echo the gold answer (--dry-run only)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log verbosity")
    return parser.parse_args(argv)


def validate_args(parser_error, args: argparse.Namespace) -> None:
    if args.n_samples < 1:
        parser_error("--n-samples must be >= 1")
    if args.threads < 1:
        parser_error("--threads must be >= 1")
    if args.max_tokens < 1:
        parser_error("--max-tokens must be >= 1")
    if args.temperature < 0:
        parser_error("--temperature must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser_error("--limit must be >= 1")
    if not 0.0 <= args.dry_run_correct_rate <= 1.0:
        parser_error("--dry-run-correct-rate must be between 0 and 1")
    if not args.dry_run and not args.model:
        parser_error("--model is required unless --dry-run is given")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per API call otherwise
    validate_args(lambda msg: sys.exit(f"DA.py: error: {msg}"), args)
    try:
        return run(args)
    except (dataio.DatasetFormatError, config.MissingCredential) as exc:
        # Configuration and input problems get a message, not a traceback.
        print(f"DA.py: error: {exc}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    items, meta = dataio.load_instances(args.data)
    items = [dataio.normalize_instance(item) for item in items]
    if args.limit:
        items = items[: args.limit]
    LOGGER.info("Loaded %d instances from %s (level=%s, year=%s)",
                len(items), args.data, meta.get("level"), meta.get("year"))

    if args.dry_run:
        model: Any = StubModel(correct_rate=args.dry_run_correct_rate)
        model_name = args.model or StubModel.name
        LOGGER.warning("DRY RUN: answers come from a stub model, scores are synthetic")
    else:
        base_url, api_key = config.openai_credentials(base_url=args.base_url, api_key=args.api_key)
        model_name = args.model
        model = ChatModel(name=model_name, base_url=base_url, api_key=api_key,
                          max_retries=args.max_retries)
        LOGGER.info("Model %s via %s", model_name, base_url)

    if args.n_samples > 1 and args.temperature == 0:
        LOGGER.warning("--n-samples %d with --temperature 0 will produce identical samples",
                       args.n_samples)

    partial_path = Path(args.partial_file) if args.partial_file else default_partial_path(
        output_dir=args.output_dir, meta=meta, model_name=model_name)
    dataio.ensure_parent(partial_path)
    # Fail before any model call if the results directory is not writable.
    results_dir = args.output_dir or os.path.join(
        "outputs", "evaluations", str(meta.get("year") or "unknown"))
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    done: Dict[str, Dict[str, Any]] = {}
    if args.resume:
        done = load_partial(partial_path)
        done = {k: v for k, v in done.items() if v.get("n_samples", 1) >= args.n_samples}
        LOGGER.info("Resuming: %d of %d questions already answered in %s",
                    len(done), len(items), partial_path)
    elif partial_path.is_file():
        LOGGER.info("Overwriting existing sidecar %s (pass --resume to keep it)", partial_path)

    pending = [(idx, item) for idx, item in enumerate(items)
               if record_key(idx, _question_of(item)) not in done]
    LOGGER.info("Running %d question(s) with %d thread(s), %d sample(s) each",
                len(pending), args.threads, args.n_samples)

    guard = FirstFailureGuard("model")
    started = time.time()
    interrupted = False
    with JsonlWriter(partial_path, append=args.resume) as writer:
        def worker(pair):
            index, item = pair
            record = process_item(index, item, model=model, guard=guard,
                                  n_samples=args.n_samples, temperature=args.temperature,
                                  max_tokens=args.max_tokens, total=len(items),
                                  dry_run=args.dry_run)
            writer.write(record)
            return record

        try:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                for record in pool.map(worker, pending):
                    done[record_key(record["index"], record["question"])] = record
        except AbortRun as exc:
            LOGGER.error("Aborting run: %s", exc)
        except KeyboardInterrupt:
            interrupted = True
            LOGGER.warning("Interrupted; %d record(s) are safe in %s (rerun with --resume)",
                           len(done), partial_path)
    elapsed = time.time() - started

    records = sorted(done.values(), key=lambda r: (r.get("index") is None, r.get("index")))
    if not records:
        LOGGER.error("No questions were answered. %s",
                     guard.reason or "Check the log above for the cause.")
        return 2

    flat = flatten_samples(records)
    summary = summarize(records=records, flat=flat, args=args, model_name=model_name,
                        elapsed=elapsed, partial_path=partial_path)
    if guard.abort.is_set():
        summary["aborted"] = True
        summary["abort_reason"] = guard.reason
    if interrupted:
        summary["interrupted"] = True

    paths = dataio.save_run(results=flat, summary=summary, method=METHOD,
                            model_name=model_name, data_path=args.data,
                            output_dir=args.output_dir, metadata=meta)
    print_summary(summary, paths)
    if guard.abort.is_set():
        print(f"\nRun aborted after the first model call failed: {guard.reason}")
        return 2
    return 130 if interrupted else 0


def _question_of(item: Dict[str, Any]) -> str:
    question = str(item.get("question", "")).strip()
    return question if question.endswith("?") else question + "?"


if __name__ == "__main__":
    sys.exit(main())
