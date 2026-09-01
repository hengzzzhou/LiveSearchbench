#!/usr/bin/env python3
"""Retrieval-augmented (iterative search) runner.

Loop: the model reasons inside ``<think>``, asks for a web search with
``<search>query</search>``, receives the top hits inside ``<information>``, and
repeats until it emits ``<answer>`` or the ``--max-iter`` search budget runs out.

Changes relative to the first release:

* ``SERPER_ENDPOINT`` used to be an empty string with no way to override it, so
  the script could not run at all. The default is now
  :data:`livesearchbench.config.DEFAULT_SERPER_ENDPOINT` and ``--serper-endpoint``
  overrides it.
* Credentials resolve through :mod:`livesearchbench.config` (``--serper-key``,
  ``--api-key``, ``--base-url`` > environment > ``.env``) instead of empty
  module-level placeholders.
* ``search()`` used to catch every exception and return ``""``, which made a bad
  key, a wrong endpoint, an exhausted quota and a network outage
  indistinguishable from "this query has no hits" -- the run then finished with a
  silently depressed score. Failures now raise, are counted per item and in the
  summary, and a failure of the very *first* search aborts the run, because that
  is always a configuration problem.
* Datasets load through :func:`livesearchbench.dataio.load_instances`, so the
  bare-list ``demo.json`` works as well as the ``{"qa_pairs": ...}`` splits.
* Scoring uses :func:`livesearchbench.scoring.score_item`; every record now
  carries ``exact_match``, ``token_f1`` and ``contains_match``. ``is_correct``
  is kept for backward compatibility but is now **normalised exact match**,
  where it used to be the much more permissive substring containment (now
  reported as ``contains_match``). Numbers from this runner are therefore not
  directly comparable with numbers printed by the first release.
* Records stream to a JSONL sidecar, so an interrupted run keeps its work and
  can be continued with ``--resume``.
* ``--n-samples N`` draws N independent search trajectories per question and
  records all of them, so ``scripts/analysis/score.py`` can compute pass@k.
* The summary sets ``max_search_calls_allowed``, so the budget is embedded in
  the output filename as ``_maxiter_<N>_`` and
  ``scripts/analysis/summarize_rag_budget.py`` can recover it.
* ``--dry-run`` swaps in an offline stub model and stub search engine.

Examples:
    python scripts/eval/RAG.py demo.json --model gpt-4o-mini --serper-key $SERPER_API_KEY
    python scripts/eval/RAG.py bench/2025/level3.json --model gpt-4o-mini --max-iter 5
    python scripts/eval/RAG.py demo.json --dry-run
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from livesearchbench import config, dataio, http, scoring  # noqa: E402

LOGGER = logging.getLogger("livesearchbench.eval.RAG")

METHOD = "RAG"
PROMPT_TEMPLATE = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> every time "
    "you get new information. If you lack knowledge, call search by <search> query </search>; you "
    "will receive results enclosed by <information> </information>. Search as many times as needed. "
    "When sufficient knowledge is gathered, output ONLY the final answer inside <answer> </answer> "
    "without extra explanation. For example, <answer> Beijing </answer>. Question: {question}"
)
NO_RESULTS_TEXT = "No search results were returned for this query."


class AbortRun(RuntimeError):
    """Raised inside a worker once the run has been declared unrecoverable."""


class SearchError(RuntimeError):
    """A search call failed. Never conflated with an empty result set."""


class FirstFailureGuard:
    """Abort the whole run when the very first call of a kind fails.

    A failure on the first call is almost always a configuration problem (wrong
    key, wrong endpoint, exhausted quota) rather than a transient one, so it is
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


class SerperSearch:
    """Serper web search with bounded retries and loud, typed failures."""

    def __init__(self, *, endpoint: str, api_key: str, top_docs: int = 3,
                 max_attempts: int = 5) -> None:
        self.endpoint = endpoint
        self.api_key = api_key
        self.top_docs = top_docs
        self._session = http.PoliteSession(component="LiveSearchBench-RAG",
                                           max_attempts=max_attempts, min_interval=0.0)
        self.calls = 0
        self.failures = 0
        self._lock = threading.Lock()

    def __call__(self, query: str, **_: Any) -> Tuple[str, int]:
        """Return ``(formatted_docs, n_docs)``; raise :class:`SearchError` on failure."""
        with self._lock:
            self.calls += 1
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        try:
            response = self._session.post(self.endpoint, headers=headers,
                                          data=json.dumps({"q": query}))
        except http.RequestFailed as exc:
            raise SearchError(f"Serper request failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise SearchError(
                f"Serper rejected the credentials (HTTP {response.status_code}). "
                f"Check SERPER_API_KEY / --serper-key."
            )
        if response.status_code == 402:
            raise SearchError("Serper reports the account is out of credits (HTTP 402).")
        if response.status_code != 200:
            raise SearchError(
                f"Serper returned HTTP {response.status_code} for {self.endpoint}: "
                f"{response.text[:200]!r}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise SearchError(
                f"Serper returned a non-JSON body ({response.text[:200]!r}). "
                f"Is --serper-endpoint correct? Default: {config.DEFAULT_SERPER_ENDPOINT}"
            ) from exc

        organic = payload.get("organic")
        if organic is None:
            # A well-formed no-hits response still carries 'organic': []. A body
            # without the key at all means the endpoint is not what we expect.
            raise SearchError(
                f"Serper response has no 'organic' field (keys: {sorted(payload)[:8]}). "
                f"Is --serper-endpoint correct?"
            )
        docs = [
            f"Doc {i + 1}(Title: {doc.get('title', '')}) {doc.get('snippet', '')}"
            for i, doc in enumerate(organic[: self.top_docs])
        ]
        return ("\n".join(docs) if docs else NO_RESULTS_TEXT), len(docs)

    def note_failure(self) -> None:
        with self._lock:
            self.failures += 1

    def close(self) -> None:
        self._session.close()


class StubSearch:
    """Offline stand-in for Serper, used by ``--dry-run``."""

    def __init__(self, *, top_docs: int = 3) -> None:
        self.top_docs = top_docs
        self.calls = 0
        self.failures = 0
        self._lock = threading.Lock()

    def __call__(self, query: str, *, gold: str = "", **_: Any) -> Tuple[str, int]:
        with self._lock:
            self.calls += 1
        docs = [f"Doc {i + 1}(Title: Stub result {i + 1} for {query!r}) "
                f"Offline dry-run snippet; no network call was made."
                for i in range(self.top_docs)]
        return "\n".join(docs), len(docs)

    def note_failure(self) -> None:
        with self._lock:
            self.failures += 1

    def close(self) -> None:
        return None


class StubModel:
    """Offline stand-in for the chat API, used by ``--dry-run``.

    It issues one search turn and then answers. Answers are synthetic: a
    deterministic, seeded fraction echo the gold answer and the rest return a
    fixed canned string. Scores produced in dry-run mode are meaningless and
    every record is tagged ``"dry_run": true``.
    """

    name = "dry-run-stub"

    def __init__(self, *, correct_rate: float = 0.5, canned: str = "Beijing",
                 searches: int = 1) -> None:
        self.correct_rate = correct_rate
        self.canned = canned
        self.searches = searches

    def complete(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        gold: str = "",
        question: str = "",
        index: int = 0,
        sample_index: int = 0,
        **_: Any,
    ) -> str:
        # messages[0] is the instruction, which mentions <information> literally.
        seen = sum(1 for m in messages[1:] if "<information>" in str(m.get("content", "")))
        if seen < self.searches:
            return (f"<think>I do not know this offhand, searching.</think>\n"
                    f"<search>{question or 'livesearchbench stub query'}</search>")
        rng = random.Random(f"{index}:{sample_index}")
        answer = str(gold) if rng.random() < self.correct_rate else self.canned
        return f"<think>The retrieved documents settle it.</think>\n<answer> {answer} </answer>"


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
        max_tokens: int = 4096,
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


def get_query(text: str) -> Optional[str]:
    """Return the last ``<search>`` query in a completion, if any."""
    matches = re.findall(r"<search>(.*?)</search>", text or "", re.DOTALL)
    return matches[-1].strip() if matches else None


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



def run_fingerprint(**parts) -> str:
    """Short stable hash of the settings that make two runs incomparable.

    Resuming used to match on level, method and model only, so changing the
    temperature, the sample count or even the dataset silently reused the old
    answers while the summary reported the new settings.
    """
    import hashlib
    payload = "\x1f".join(f"{k}={parts[k]!r}" for k in sorted(parts))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=5).hexdigest()

def default_partial_path(
    *,
    output_dir: Optional[str],
    meta: Dict[str, Any],
    model_name: str,
    budget: Optional[int] = None,
    fingerprint: str = "",
) -> Path:
    """Deterministic sidecar path, so ``--resume`` can find a previous run.

    ``fingerprint`` separates runs whose settings differ, so ``--resume`` can
    only ever continue a run that used the same configuration.
    """
    level = meta.get("level") or "unknown"
    year = meta.get("year") or "unknown"
    tag = f"_maxiter_{budget}" if budget is not None else ""
    fp = f"_{fingerprint}" if fingerprint else ""
    directory = Path(output_dir or os.path.join("outputs", "evaluations", str(year)))
    return directory / f"{level}_{METHOD}_{safe_model_tag(model_name)}{tag}{fp}_partial.jsonl"


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


def run_trajectory(
    *,
    question: str,
    gold: str,
    aliases: Sequence[str],
    index: int,
    sample_index: int,
    model: Any,
    search: Any,
    model_guard: FirstFailureGuard,
    search_guard: FirstFailureGuard,
    max_iterations: int,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """One search-and-answer trajectory. Returns the sample record."""
    prompt = PROMPT_TEMPLATE.format(question=question)
    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    trace: List[Dict[str, Any]] = [{"type": "initial_prompt", "content": prompt}]
    search_count = 0
    search_failures = 0
    search_errors: List[str] = []
    answer = ""
    error: Optional[str] = None
    model_failed = False

    # Budget search turns, plus one turn to answer and one after the budget notice.
    for step in range(max_iterations + 2):
        model_guard.check()
        search_guard.check()
        try:
            content = model.complete(
                messages, max_tokens=max_tokens, temperature=temperature,
                gold=gold, question=question, index=index, sample_index=sample_index,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never silently dropped
            model_guard.record(False, str(exc))
            error = f"{type(exc).__name__}: {exc}"
            LOGGER.error("Item %d sample %d: model call failed: %s", index, sample_index, error)
            trace.append({"type": "error", "step": step, "content": error})
            model_failed = True
            break
        model_guard.record(True)
        trace.append({"type": "model_output", "step": step, "content": content})

        query = get_query(content)
        if query is None:
            trace.append({"type": "final_output", "content": content})
            answer = extract_answer(content)
            break

        if search_count >= max_iterations:
            # Budget exhausted: tell the model instead of silently looping.
            trace.append({"type": "budget_exhausted", "step": step,
                          "content": f"search budget of {max_iterations} calls is spent"})
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                             "<information>The search budget is exhausted. "
                             "Answer now inside <answer> </answer>.</information>"})
            continue

        try:
            docs, n_docs = search(query, gold=gold)
        except SearchError as exc:
            search_guard.record(False, str(exc))
            search_failures += 1
            search.note_failure()
            search_errors.append(str(exc))
            LOGGER.error("Item %d sample %d: search failed: %s", index, sample_index, exc)
            docs, n_docs = f"Search failed: {exc}", 0
            trace.append({"type": "search_error", "step": step, "query": query, "content": str(exc)})
        else:
            search_guard.record(True)
            trace.append({"type": "search_results", "step": step,
                          "content": docs, "n_docs": n_docs})
        search_count += 1
        trace.append({"type": "search_query", "step": step, "query": query})
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"<information>{docs}</information>"})
    else:
        error = error or f"Maximum search iterations ({max_iterations}) reached without an answer"
        trace.append({"type": "error", "content": error})

    scores = scoring.score_item(answer, gold, aliases)
    sample = {
        "sample_index": sample_index,
        "model_answer": answer,
        "search_count": search_count,
        "search_failures": search_failures,
        "reasoning_process": trace,
        **scores,
    }
    if search_errors:
        sample["search_errors"] = search_errors
    if error:
        sample["error"] = error
    if model_failed:
        sample["model_failed"] = True
    return sample


def process_item(
    index: int,
    item: Dict[str, Any],
    *,
    model: Any,
    search: Any,
    model_guard: FirstFailureGuard,
    search_guard: FirstFailureGuard,
    n_samples: int,
    max_iterations: int,
    temperature: float,
    max_tokens: int,
    total: int,
    dry_run: bool,
) -> Dict[str, Any]:
    model_guard.check()
    search_guard.check()
    question = str(item.get("question", "")).strip()
    if not question.endswith("?"):
        question += "?"
    gold = item.get("answer", "")
    aliases = item.get("answer_aliases") or []

    samples: List[Dict[str, Any]] = []
    for sample_index in range(n_samples):
        sample = run_trajectory(
            question=question, gold=gold, aliases=aliases, index=index,
            sample_index=sample_index, model=model, search=search,
            model_guard=model_guard, search_guard=search_guard,
            max_iterations=max_iterations, temperature=temperature, max_tokens=max_tokens,
        )
        samples.append(sample)
        if sample.get("model_failed"):
            # The endpoint is broken; drawing the remaining samples cannot help.
            break

    primary = samples[0]
    record = {
        "index": index,
        "question": question,
        "expected_answer": gold,
        "answer_aliases": aliases,
        "level": item.get("level"),
        "model_answer": primary["model_answer"],
        "is_correct": bool(primary["exact_match"]),
        "exact_match": primary["exact_match"],
        "token_f1": primary["token_f1"],
        "contains_match": primary["contains_match"],
        "search_count": primary["search_count"],
        "search_failures": sum(s["search_failures"] for s in samples),
        "total_search_count": sum(s["search_count"] for s in samples),
        "n_samples": len(samples),
        "n_correct": int(sum(s["exact_match"] for s in samples)),
        "samples": samples,
        "reasoning_process": primary["reasoning_process"],
    }
    errors = [s["error"] for s in samples if s.get("error")]
    if errors:
        record["error"] = errors[0]
    if dry_run:
        record["dry_run"] = True
    LOGGER.info("[%d/%d] EM=%d F1=%.2f searches=%d | gold=%r | pred=%r",
                index + 1, total, int(record["exact_match"]), record["token_f1"],
                record["search_count"], gold, record["model_answer"])
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
                "search_count": sample.get("search_count", 0),
                "search_failures": sample.get("search_failures", 0),
                "reasoning_process": sample.get("reasoning_process", []),
            }
            for key in ("search_errors", "error"):
                if key in sample:
                    row[key] = sample[key]
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
    search: Any,
    elapsed: float,
    partial_path: Path,
) -> Dict[str, Any]:
    """Summarise a run.

    Metrics are computed over every sample, so with ``--n-samples > 1`` they are
    sample-averaged rather than per-question; pass@k is reported separately.
    """
    total = len(records)
    correct = sum(1 for r in flat if r.get("is_correct"))
    searches = [int(r.get("search_count", 0) or 0) for r in flat]
    failed_items = sum(1 for r in records if r.get("search_failures"))
    summary: Dict[str, Any] = {
        "method": METHOD,
        "model": model_name,
        "search_enabled": True,
        "total_questions": total,
        "total_samples": len(flat),
        "correct_answers": correct,
        "accuracy": round(correct / len(flat), 4) if flat else 0.0,
        "max_search_calls_allowed": args.max_iter,
        "avg_search_calls": round(sum(searches) / len(searches), 2) if searches else 0.0,
        "max_search_calls_used": max(searches) if searches else 0,
        "zero_search_trajectories": sum(1 for c in searches if c == 0),
        "search_calls_attempted": getattr(search, "calls", 0),
        "search_calls_failed": getattr(search, "failures", 0),
        "items_with_search_failure": failed_items,
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "serper_endpoint": getattr(search, "endpoint", "stub"),
        "metrics": scoring.aggregate(flat, group_key="level"),
        "items_with_errors": sum(1 for r in records if r.get("error")),
        "elapsed_seconds": round(elapsed, 1),
        "test_file": args.data,
        "partial_file": str(partial_path),
        "dry_run": bool(args.dry_run),
    }
    if args.n_samples > 1 and total:
        summary["pass_at_k"] = {
            f"pass@{k}": round(
                100.0 * sum(
                    scoring.pass_at_k(r.get("n_samples", 1), r.get("n_correct", 0), k)
                    for r in records
                ) / total, 2)
            for k in range(1, args.n_samples + 1)
        }
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
    for level, stat in (summary["metrics"].get("by_level") or {}).items():
        print(f"  level {level:<10}: EM {stat['exact_match']['value']:6.2f}  "
              f"F1 {stat['token_f1']['value']:6.2f}  n={stat['n']}")
    if summary.get("pass_at_k"):
        print("pass@k           : " + ", ".join(f"{k}={v:.2f}" for k, v in summary["pass_at_k"].items()))
    print(f"Search budget    : {summary['max_search_calls_allowed']} call(s) per trajectory")
    print(f"Search calls     : {summary['search_calls_attempted']} attempted, "
          f"{summary['search_calls_failed']} failed "
          f"({summary['items_with_search_failure']} item(s) affected)")
    print(f"Avg calls/traj.  : {summary['avg_search_calls']:.2f} "
          f"({summary['zero_search_trajectories']} trajectory/-ies never searched)")
    if summary["search_calls_failed"]:
        print("WARNING: some searches failed. Accuracy below is a lower bound; "
              "see 'search_errors' in the result records.")
    if summary.get("items_with_errors"):
        print(f"Items with errors: {summary['items_with_errors']}")
    if summary.get("dry_run"):
        print("NOTE: --dry-run was used. These numbers come from a stub model and mean nothing.")
    print(f"Results          : {paths['results']}")
    print(f"Summary          : {paths['summary']}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="RAG.py",
        description="Iterative-search (RAG) runner for LiveSearchBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("data", help="Benchmark split, e.g. demo.json or bench/2025/level3.json")
    parser.add_argument("--model", default=None,
                        help="Model name sent to the OpenAI-compatible endpoint "
                             "(required unless --dry-run)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL; falls back to $OPENAI_BASE_URL, then .env, "
                             "then https://api.openai.com/v1")
    parser.add_argument("--api-key", default=None,
                        help="API key; falls back to $OPENAI_API_KEY, then .env")
    parser.add_argument("--serper-key", default=None,
                        help="Serper API key; falls back to $SERPER_API_KEY, then .env")
    parser.add_argument("--serper-endpoint", default=None,
                        help=f"Serper search endpoint; falls back to $SERPER_ENDPOINT, then .env, "
                             f"then {config.DEFAULT_SERPER_ENDPOINT}")
    parser.add_argument("--top-docs", type=int, default=3,
                        help="Organic results injected into each <information> block")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="Search budget: maximum search calls per trajectory")
    parser.add_argument("--threads", type=int, default=4, help="Questions answered in parallel")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Generation cap per call")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Trajectories per question; >1 records every sample so that "
                             "scripts/analysis/score.py can compute pass@k")
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Bounded retries per model call before the item is marked failed")
    parser.add_argument("--search-retries", type=int, default=5,
                        help="Bounded retries per search call before it is counted as failed")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions")
    parser.add_argument("--output-dir", default=None,
                        help="Where results land; default outputs/evaluations/<year>")
    parser.add_argument("--partial-file", default=None,
                        help="JSONL sidecar written as the run proceeds; default is derived from "
                             "the split, model name and budget so --resume can find it")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse records already present in the sidecar and only run what is missing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use an offline stub model and stub search engine; needs no credentials")
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
    if args.max_iter < 0:
        parser_error("--max-iter must be >= 0")
    if args.top_docs < 1:
        parser_error("--top-docs must be >= 1")
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
    validate_args(lambda msg: sys.exit(f"RAG.py: error: {msg}"), args)
    try:
        return run(args)
    except (dataio.DatasetFormatError, config.MissingCredential) as exc:
        # Configuration and input problems get a message, not a traceback.
        print(f"RAG.py: error: {exc}", file=sys.stderr)
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
        search: Any = StubSearch(top_docs=args.top_docs)
        LOGGER.warning("DRY RUN: answers and search results come from stubs, scores are synthetic")
    else:
        base_url, api_key = config.openai_credentials(base_url=args.base_url, api_key=args.api_key)
        endpoint, serper_key = config.serper_credentials(endpoint=args.serper_endpoint,
                                                         api_key=args.serper_key)
        model_name = args.model
        model = ChatModel(name=model_name, base_url=base_url, api_key=api_key,
                          max_retries=args.max_retries)
        search = SerperSearch(endpoint=endpoint, api_key=serper_key, top_docs=args.top_docs,
                              max_attempts=args.search_retries)
        LOGGER.info("Model %s via %s; search via %s", model_name, base_url, endpoint)

    if args.n_samples > 1 and args.temperature == 0:
        LOGGER.warning("--n-samples %d with --temperature 0 will produce identical samples",
                       args.n_samples)

    fingerprint = run_fingerprint(dataset=str(args.data), model=model_name, temperature=args.temperature,


        n_samples=args.n_samples, max_tokens=args.max_tokens, method=METHOD, max_iter=args.max_iter)


    partial_path = Path(args.partial_file) if args.partial_file else default_partial_path(


        output_dir=args.output_dir, meta=meta, model_name=model_name,


        fingerprint=fingerprint, budget=args.max_iter)
    dataio.ensure_parent(partial_path)
    # Fail before any model or search call if the results directory is not writable.
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
    LOGGER.info("Running %d question(s) with %d thread(s), %d sample(s) each, budget %d search call(s)",
                len(pending), args.threads, args.n_samples, args.max_iter)

    model_guard = FirstFailureGuard("model")
    search_guard = FirstFailureGuard("search")
    started = time.time()
    interrupted = False
    with JsonlWriter(partial_path, append=args.resume) as writer:
        def worker(pair):
            index, item = pair
            record = process_item(index, item, model=model, search=search,
                                  model_guard=model_guard, search_guard=search_guard,
                                  n_samples=args.n_samples, max_iterations=args.max_iter,
                                  temperature=args.temperature, max_tokens=args.max_tokens,
                                  total=len(items), dry_run=args.dry_run)
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
    search.close()

    records = sorted(done.values(), key=lambda r: (r.get("index") is None, r.get("index")))
    if not records:
        reason = search_guard.reason or model_guard.reason or "Check the log above for the cause."
        LOGGER.error("No questions were answered. %s", reason)
        return 2

    flat = flatten_samples(records)
    summary = summarize(records=records, flat=flat, args=args, model_name=model_name,
                        search=search, elapsed=elapsed, partial_path=partial_path)
    aborted = search_guard.abort.is_set() or model_guard.abort.is_set()
    if aborted:
        summary["aborted"] = True
        summary["abort_reason"] = search_guard.reason or model_guard.reason
    if interrupted:
        summary["interrupted"] = True

    paths = dataio.save_run(results=flat, summary=summary, method=METHOD,
                            model_name=model_name, data_path=args.data,
                            output_dir=args.output_dir, metadata=meta)
    print_summary(summary, paths)
    if aborted:
        print(f"\nRun aborted after the first {'search' if search_guard.abort.is_set() else 'model'} "
              f"call failed: {summary['abort_reason']}")
        return 2
    return 130 if interrupted else 0


def _question_of(item: Dict[str, Any]) -> str:
    question = str(item.get("question", "")).strip()
    return question if question.endswith("?") else question + "?"


if __name__ == "__main__":
    sys.exit(main())
