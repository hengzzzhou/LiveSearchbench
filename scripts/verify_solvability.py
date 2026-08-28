#!/usr/bin/env python3
"""Step 4 of the LiveSearchBench pipeline: Contextual Solvability Verification.

After a question has been synthesised from a Wikidata change, it is handed to a
verifier model together with the *gold context* -- the provenance the question
was generated from -- and nothing else. No retrieval, no web search, no reliance
on the model's own memory. An instance is retained only when the verifier
recovers the gold answer from that context alone.

The stage exists to catch two failure modes that survive SPARQL verification:

* **ambiguity** -- the question, read literally, does not pin down a single
  entity even when the underlying triple does;
* **generation hallucination** -- the synthesiser wrote a question about
  something the source triple does not actually say.

Gold context is assembled per level from whatever provenance that level carries:

===========  ============================================================
level        provenance used
===========  ============================================================
1            ``source_triple`` (subject / property / value, with QIDs)
2            ``constraint_info`` plus ``reasoning_chain``
3            ``abstraction_info`` + ``original_level2_question`` + chain
any          ``reasoning_chain`` alone (fallback)
none         reported explicitly as ``context_source: none``
===========  ============================================================

Instances with no usable provenance (``demo.json`` carries none) are never
silently passed: ``--on-missing-context`` decides whether they are rejected
(default), kept, or set aside as unverifiable. ``--resolve-sparql`` can
reconstruct a weak context for them by running their ``sparql_verification``
program through WDQS and labelling the entities it returns.

Outputs
    * a per-instance verdict file -- kept/rejected, the verifier's answers, and
      exactly which context it saw;
    * a filtered copy of the split holding only the instances that passed, in
      the same JSON shape as the input, so this is a usable pipeline filter and
      not merely a report.

Examples
    # offline smoke test, no API key needed
    python scripts/verify_solvability.py demo.json --dry-run --resolve-sparql

    # the real thing
    python scripts/verify_solvability.py bench/2025/level2.json \\
        --model Qwen3-235B-A22B-Instruct-2507 --n-votes 3 --concurrency 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from livesearchbench import config, dataio, scoring  # noqa: E402

LOGGER = logging.getLogger("verify_solvability")

#: The verifier named in the paper. Any OpenAI-compatible endpoint will do.
DEFAULT_MODEL = "Qwen3-235B-A22B-Instruct-2507"

#: Sentinel the verifier is told to emit when the gold context is insufficient.
INSUFFICIENT = "NOT_IN_CONTEXT"

VERIFIER_SYSTEM_PROMPT = (
    "You are a strict verification model. You answer only from the context you are given. "
    "You have no access to any search tool, and you must not fall back on your own world "
    "knowledge. Answering from memory defeats the purpose of this task."
)

#: The Step-4 prompt. Kept as a module-level constant so it can be grepped,
#: diffed, and quoted verbatim in the paper's appendix.
VERIFIER_PROMPT = """You are checking whether a benchmark question can be answered from its own gold context.

The GOLD CONTEXT below is the Wikidata provenance the question was generated from.
It is the only information you are permitted to use.

Rules:
1. Use ONLY the gold context. Do not use outside knowledge, memorised facts, or guesses.
2. You have no retrieval tool and must not pretend to have consulted one.
3. If the gold context does not single out one unambiguous answer to the question as
   the question is literally written, reply with exactly {insufficient}.
4. Reply with the answer entity alone: no explanation, no sentence, no restatement.
5. Wrap the reply in <answer> and </answer> tags, for example
   <answer>Beijing</answer> or <answer>{insufficient}</answer>.

--- BEGIN GOLD CONTEXT ---
{gold_context}
--- END GOLD CONTEXT ---

QUESTION: {question}
"""


class VerifierError(RuntimeError):
    """Raised when the verifier endpoint could not be reached or parsed."""


# ---------------------------------------------------------------------------
# Gold-context construction
# ---------------------------------------------------------------------------


@dataclass
class GoldContext:
    """A block of provenance text plus the field it was derived from."""

    text: str
    source: str

    @property
    def usable(self) -> bool:
        return self.source != "none" and bool(self.text.strip())


_EMPTY_CONTEXT = GoldContext("", "none")


def _format_triples(chain: Any) -> List[str]:
    """Render a ``reasoning_chain`` as one bullet per triple."""
    lines: List[str] = []
    for triple in chain or []:
        if isinstance(triple, (list, tuple)) and len(triple) >= 3:
            lines.append(f"  - ({triple[0]}, {triple[1]}, {triple[2]})")
        elif isinstance(triple, dict):
            subj = triple.get("subject_label") or triple.get("subject") or "?"
            pred = triple.get("predicate_label") or triple.get("predicate") or "?"
            obj = triple.get("object_label") or triple.get("object") or "?"
            lines.append(f"  - ({subj}, {pred}, {obj})")
        elif triple:
            lines.append(f"  - {triple}")
    return lines


def _bullets(values: Any) -> List[str]:
    out: List[str] = []
    for value in values or []:
        text = str(value).strip()
        if text:
            out.append(f"  - {text}")
    return out


def build_level1_context(item: Dict[str, Any]) -> Optional[GoldContext]:
    """Level 1: the single source statement the question was written from."""
    triple = item.get("source_triple")
    if not isinstance(triple, dict):
        return None
    subject = triple.get("subject_label")
    predicate = triple.get("predicate_label")
    if not subject or not predicate:
        return None
    value = triple.get("object_label") or triple.get("object_value") or triple.get("object_id")
    if not value:
        return None

    def _with_id(label: Any, ident: Any) -> str:
        return f"{label} ({ident})" if ident else str(label)

    lines = [
        "Source Wikidata statement:",
        f"  subject:  {_with_id(subject, triple.get('subject_id'))}",
        f"  property: {_with_id(predicate, triple.get('predicate_id'))}",
        f"  value:    {_with_id(value, triple.get('object_id'))}",
    ]
    return GoldContext("\n".join(lines), "source_triple")


def build_level2_context(item: Dict[str, Any]) -> Optional[GoldContext]:
    """Level 2: the constraint set the answer entity must jointly satisfy."""
    info = item.get("constraint_info")
    if not isinstance(info, dict):
        return None
    constraints = _bullets(info.get("constraints"))
    if not constraints:
        return None
    lines = ["Constraints the answer entity satisfies:"] + constraints
    if info.get("constraint_count") is not None:
        lines.append(f"  (all {info['constraint_count']} constraints must hold simultaneously)")
    triples = _format_triples(item.get("reasoning_chain"))
    if triples:
        lines.append("Supporting Wikidata statements:")
        lines.extend(triples)
    return GoldContext("\n".join(lines), "constraint_info")


def build_level3_context(item: Dict[str, Any]) -> Optional[GoldContext]:
    """Level 3: the abstraction map plus the unabstracted Level 2 question."""
    info = item.get("abstraction_info")
    original = str(item.get("original_level2_question") or "").strip()
    if not isinstance(info, dict) and not original:
        return None
    lines: List[str] = []
    if isinstance(info, dict):
        abstract = _bullets(info.get("abstract_constraints"))
        concrete = _bullets(info.get("original_constraints"))
        if abstract:
            lines.append("Indirect constraints as phrased in the question:")
            lines.extend(abstract)
        if concrete:
            lines.append("The concrete Wikidata constraints they stand for:")
            lines.extend(concrete)
    if original:
        lines.append("The same question before abstraction:")
        lines.append(f"  {original}")
    triples = _format_triples(item.get("reasoning_chain"))
    if triples:
        lines.append("Supporting Wikidata statements:")
        lines.extend(triples)
    if not lines:
        return None
    return GoldContext("\n".join(lines), "abstraction_info")


def build_chain_context(item: Dict[str, Any]) -> Optional[GoldContext]:
    """Fallback: the reasoning chain on its own."""
    triples = _format_triples(item.get("reasoning_chain"))
    if not triples:
        return None
    return GoldContext("\n".join(["Supporting Wikidata statements:"] + triples), "reasoning_chain")


_LEVEL_BUILDERS: Dict[str, Callable[[Dict[str, Any]], Optional[GoldContext]]] = {
    "1": build_level1_context,
    "2": build_level2_context,
    "3": build_level3_context,
}


def build_gold_context(
    item: Dict[str, Any],
    *,
    resolver: Optional["SparqlContextResolver"] = None,
) -> GoldContext:
    """Assemble the gold context for one instance.

    The builder matching the instance's declared level is tried first, then the
    remaining builders, then the reasoning-chain fallback, then the optional
    SPARQL resolver. Returns :data:`_EMPTY_CONTEXT` when nothing is available,
    which the caller must report rather than treat as an empty success.
    """
    level = str(item.get("level") or "").strip()
    ordered: List[Callable[[Dict[str, Any]], Optional[GoldContext]]] = []
    if level in _LEVEL_BUILDERS:
        ordered.append(_LEVEL_BUILDERS[level])
    for key, builder in _LEVEL_BUILDERS.items():
        if key != level:
            ordered.append(builder)
    ordered.append(build_chain_context)

    for builder in ordered:
        context = builder(item)
        if context is not None and context.usable:
            return context

    if resolver is not None:
        context = resolver.resolve(item)
        if context is not None and context.usable:
            return context
    return _EMPTY_CONTEXT


class SparqlContextResolver:
    """Reconstruct a weak gold context from ``sparql_verification`` via WDQS.

    The COUNT program is rewritten into a labelled SELECT and executed; the
    labels it returns become the context. This is weaker provenance than a
    source triple -- it names the answer set directly instead of the statement
    the question was written from -- so it is opt-in, and every instance that
    uses it is tagged ``context_source: sparql_labels`` in the verdict file.
    It also reflects *today's* Wikidata, which for an older split may differ
    from the state the question was generated against.
    """

    def __init__(self, *, lang: str = "en", min_interval: float = 1.0) -> None:
        from livesearchbench import sparql as sparql_mod  # local: only needed with --resolve-sparql

        self._sparql = sparql_mod
        self._client = sparql_mod.SparqlClient(min_interval=min_interval)
        self._lang = lang
        self._lock = threading.Lock()
        self.attempted = 0
        self.recovered = 0

    def resolve(self, item: Dict[str, Any]) -> Optional[GoldContext]:
        program = str(item.get("sparql_verification") or "").strip()
        if not program:
            return None
        select = self._sparql.to_label_select(program, lang=self._lang)
        if not select:
            return None
        with self._lock:  # SparqlClient rate-limits per instance; keep calls serial
            self.attempted += 1
            try:
                labels = self._client.select_labels(select, lang=self._lang)
            except self._sparql.SparqlError as exc:
                LOGGER.warning("WDQS lookup failed for %r: %s", item.get("question", "")[:60], exc)
                return None
            if labels:
                self.recovered += 1
        if not labels:
            return None
        lines = [
            "Wikidata verification program for this question:",
            *[f"  {line}" for line in program.splitlines() if line.strip()],
            "Entities that program returns, resolved to English labels:",
            *[f"  - {label}" for label in labels],
        ]
        return GoldContext("\n".join(lines), "sparql_labels")

    def close(self) -> None:
        try:
            self._client.session.close()
        except Exception:  # noqa: BLE001 - closing is best effort
            pass


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def extract_answer(text: str) -> str:
    """Pull the answer out of a verifier reply, tolerating a missing tag."""
    if not text:
        return ""
    match = _ANSWER_RE.search(text)
    if match:
        return match.group(1).strip().strip('"').strip()
    # No tag: fall back to the last non-empty line, which is where a compliant
    # model puts the answer even when it forgets the markup.
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1].strip('"').strip() if lines else ""


class OpenAIVerifier:
    """Calls an OpenAI-compatible chat endpoint with bounded retries."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        max_attempts: int = 4,
        timeout: float = 120.0,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise SystemExit(
                "The 'openai' package is required to call a verifier.\n"
                "  pip install -r requirements.txt\n"
                "  or run with --dry-run to exercise the pipeline offline."
            ) from exc
        self.model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_attempts = max(1, max_attempts)
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    def answer(self, *, question: str, gold_context: str, gold_answer: Any, vote: int) -> str:
        del gold_answer, vote  # the real verifier must never see the gold answer
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": VERIFIER_PROMPT.format(
                    gold_context=gold_context, question=question, insufficient=INSUFFICIENT
                ),
            },
        ]
        delay = 1.0
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:  # noqa: BLE001 - the SDK raises many types
                if attempt >= self._max_attempts:
                    raise VerifierError(
                        f"{self.model} failed after {self._max_attempts} attempts: {exc}"
                    ) from exc
                LOGGER.warning("verifier call failed (attempt %d/%d): %s", attempt, self._max_attempts, exc)
                time.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 30.0)
        raise VerifierError("unreachable")  # pragma: no cover


class StubVerifier:
    """Deterministic offline stand-in used by ``--dry-run``.

    It reads the gold answer -- which a real verifier never sees -- and echoes
    it, except for three reproducible buckets keyed on ``sha1(question)`` so
    that the reject path and majority voting are exercised too:

    ``bucket 0``  replies ``NOT_IN_CONTEXT``            -> rejected outright
    ``bucket 1``  replies with the gold answer's first token only
                  -> fails exact match, keeps partial token F1
    ``bucket 2``  replies ``NOT_IN_CONTEXT`` on vote 0 and the gold answer on
                  later votes -> rejected at ``--n-votes 1``, kept at 3
    """

    model = "stub"

    @staticmethod
    def bucket(question: str) -> int:
        digest = hashlib.sha1(str(question).encode("utf-8")).hexdigest()
        return int(digest, 16) % 12

    def answer(self, *, question: str, gold_context: str, gold_answer: Any, vote: int) -> str:
        del gold_context
        gold = str(gold_answer if not isinstance(gold_answer, (list, tuple)) else (gold_answer or [""])[0])
        bucket = self.bucket(question)
        if bucket == 0:
            reply = INSUFFICIENT
        elif bucket == 1:
            head = gold.split()[0] if gold.split() else ""
            reply = head if len(gold.split()) > 1 else INSUFFICIENT
        elif bucket == 2:
            reply = INSUFFICIENT if vote == 0 else gold
        else:
            reply = gold
        return f"<answer>{reply}</answer>"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def instance_key(index: int, item: Dict[str, Any]) -> str:
    """Stable identifier for --resume: position plus a hash of the question."""
    digest = hashlib.sha1(str(item.get("question", "")).encode("utf-8")).hexdigest()[:12]
    return f"{index:05d}:{digest}"


def level_name(item: Dict[str, Any]) -> str:
    level = item.get("level")
    return f"level{level}" if level is not None else "unknown"


def verify_instance(
    index: int,
    item: Dict[str, Any],
    *,
    verifier: Any,
    n_votes: int,
    on_missing: str,
    resolver: Optional[SparqlContextResolver],
) -> Dict[str, Any]:
    """Verify one instance and return its verdict record."""
    normalized = dataio.normalize_instance(item)
    question = normalized["question"]
    gold = normalized["answer"]
    aliases = normalized.get("answer_aliases") or []

    record: Dict[str, Any] = {
        "key": instance_key(index, item),
        "index": index,
        "level": item.get("level"),
        "level_name": level_name(item),
        "question": question,
        "expected_answer": gold,
        "answer_aliases": aliases,
    }

    context = build_gold_context(item, resolver=resolver)
    record["context_source"] = context.source
    record["gold_context"] = context.text

    if not context.usable:
        status = {"reject": "rejected", "keep": "kept", "skip": "unverifiable"}[on_missing]
        record.update(
            status=status,
            kept=(status == "kept"),
            reason="no_gold_context",
            note=(
                "instance carries no source_triple, constraint_info, abstraction_info "
                "or reasoning_chain; the verifier was not called"
            ),
            model_answer="",
            verifier_replies=[],
            n_votes=0,
            n_votes_exact=0,
            exact_match=0.0,
            token_f1=0.0,
            contains_match=0.0,
        )
        return record

    replies: List[str] = []
    answers: List[str] = []
    try:
        for vote in range(n_votes):
            reply = verifier.answer(
                question=question, gold_context=context.text, gold_answer=gold, vote=vote
            )
            replies.append(reply)
            answers.append(extract_answer(reply))
    except VerifierError as exc:
        LOGGER.error("instance %d: %s", index, exc)
        record.update(
            status="error",
            kept=False,
            reason="verifier_error",
            error=str(exc),
            model_answer="",
            verifier_replies=replies,
            n_votes=len(replies),
            n_votes_exact=0,
            exact_match=0.0,
            token_f1=0.0,
            contains_match=0.0,
        )
        return record

    n_exact = sum(1 for a in answers if scoring.exact_match(a, gold, aliases))
    # Strict majority: a tie at an even --n-votes rejects.
    kept = 2 * n_exact > n_votes
    counts = Counter(scoring.normalize_answer(a) for a in answers)
    modal_norm, modal_count = counts.most_common(1)[0]
    modal_answer = next((a for a in answers if scoring.normalize_answer(a) == modal_norm), answers[0])
    metrics = scoring.score_item(modal_answer, gold, aliases)

    record.update(
        status="kept" if kept else "rejected",
        kept=kept,
        reason=(
            "verified"
            if kept
            else ("insufficient_context" if INSUFFICIENT.lower() in modal_answer.lower() else "answer_mismatch")
        ),
        model_answer=modal_answer,
        verifier_answers=answers,
        verifier_replies=replies,
        n_votes=n_votes,
        n_votes_exact=n_exact,
        vote_agreement=round(modal_count / len(answers), 3),
        **metrics,
    )
    return record


# ---------------------------------------------------------------------------
# Reporting and output
# ---------------------------------------------------------------------------

_DECIDED = ("kept", "rejected")


def was_verified(record: Dict[str, Any]) -> bool:
    """True when the verifier actually ran on this instance.

    Instances with no usable provenance are decided by ``--on-missing-context``
    without a model call, so they are excluded from the pass rate. Folding them
    in would let ``--on-missing-context keep`` report a 100% pass rate for
    questions nothing ever checked.
    """
    return record.get("status") in _DECIDED and record.get("reason") != "no_gold_context"


def summarise(
    verdicts: Sequence[Dict[str, Any]], *, confidence: float, resamples: int, seed: int
) -> Dict[str, Any]:
    """Pass rate over verified instances, plus overall retention, with CIs."""
    verified = [v for v in verdicts if was_verified(v)]

    def _rate(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        values = [100.0 * float(bool(r.get("kept"))) for r in records]
        ci = scoring.bootstrap_ci(values, confidence=confidence, resamples=resamples, seed=seed)
        return {
            "n": len(records),
            "kept": sum(1 for r in records if r.get("kept")),
            "pass_rate": round(ci["mean"], 2),
            "ci_low": round(ci["lo"], 2),
            "ci_high": round(ci["hi"], 2),
        }

    by_level: Dict[str, Any] = {}
    for record in verified:
        by_level.setdefault(record.get("level_name", "unknown"), []).append(record)

    summary: Dict[str, Any] = {
        "n_instances": len(verdicts),
        "n_verified": len(verified),
        "n_no_context": sum(1 for v in verdicts if v.get("reason") == "no_gold_context"),
        "status_counts": dict(Counter(str(v.get("status")) for v in verdicts)),
        "context_sources": dict(Counter(str(v.get("context_source")) for v in verdicts)),
        "reject_reasons": dict(
            Counter(str(v.get("reason")) for v in verdicts if v.get("status") == "rejected")
        ),
        "overall": _rate(verified),
        "by_level": {name: _rate(recs) for name, recs in sorted(by_level.items())},
        "retention": _rate(verdicts),
        "confidence": confidence,
    }
    if verified:
        summary["metrics"] = scoring.aggregate(
            verified,
            prediction_key="model_answer",
            gold_key="expected_answer",
            alias_key="answer_aliases",
            group_key="level_name",
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        )
    return summary


def render_summary(summary: Dict[str, Any], *, data_path: str, model: str) -> str:
    """Human-readable report for stdout."""
    lines = [
        "",
        "=== Contextual Solvability Verification (Step 4) ===",
        f"dataset : {data_path}",
        f"verifier: {model}",
        f"instances: {summary['n_instances']}  verified by the model: {summary['n_verified']}",
        "",
        "status          : " + ", ".join(f"{k}={v}" for k, v in sorted(summary["status_counts"].items())),
        "context source  : " + ", ".join(f"{k}={v}" for k, v in sorted(summary["context_sources"].items())),
    ]
    if summary["reject_reasons"]:
        lines.append(
            "reject reasons  : " + ", ".join(f"{k}={v}" for k, v in sorted(summary["reject_reasons"].items()))
        )
    pct = int(round(100 * summary.get("confidence", 0.95)))
    overall = summary["overall"]
    lines.append("")
    if overall["n"]:
        lines.append(
            f"pass rate (verified instances): {overall['pass_rate']:.2f}%  "
            f"[{overall['ci_low']:.2f}, {overall['ci_high']:.2f}] {pct}% CI  "
            f"({overall['kept']}/{overall['n']} kept)"
        )
    else:
        why = (
            "every instance lacked usable provenance (try --resolve-sparql)"
            if summary["n_no_context"]
            else "no instance produced a verdict; see the error count above"
        )
        lines.append(f"pass rate (verified instances): n/a -- the verifier ran on nothing; {why}")
    if summary["by_level"]:
        lines.append("pass rate by level:")
        for name, stats in summary["by_level"].items():
            lines.append(
                f"  {name:<10} {stats['pass_rate']:>6.2f}%  "
                f"[{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]  "
                f"({stats['kept']}/{stats['n']})"
            )
    retention = summary["retention"]
    lines.append(
        f"retention (whole split): {retention['pass_rate']:.2f}%  "
        f"[{retention['ci_low']:.2f}, {retention['ci_high']:.2f}] {pct}% CI  "
        f"({retention['kept']}/{retention['n']} written to the filtered split)"
    )
    if summary["n_no_context"]:
        lines.append(
            f"  note: {summary['n_no_context']} instance(s) had no usable provenance and were "
            f"handled by --on-missing-context without a model call"
        )
    metrics = summary.get("metrics", {}).get("overall")
    if metrics:
        lines += [
            "",
            "verifier answer vs gold (majority answer, verified instances only):",
            "  " + "  ".join(
                f"{name}={metrics[name]['value']:.2f} "
                f"[{metrics[name]['ci_low']:.2f}, {metrics[name]['ci_high']:.2f}]"
                for name in ("exact_match", "token_f1", "contains_match")
            ),
        ]
    lines.append("")
    return "\n".join(lines)


def write_filtered(
    *,
    source_path: Path,
    items: Sequence[Dict[str, Any]],
    kept_indices: Sequence[int],
    out_path: Path,
    verification_meta: Dict[str, Any],
) -> None:
    """Write the surviving instances, preserving the input file's JSON shape."""
    kept = [items[i] for i in kept_indices]
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        item_key = next(
            (k for k in ("qa_pairs", "questions", "instances", "data") if isinstance(raw.get(k), list)),
            "qa_pairs",
        )
        meta_key = next(
            (k for k in ("metadata", "dataset_info", "info") if isinstance(raw.get(k), dict)), "metadata"
        )
        payload: Dict[str, Any] = dict(raw)
        payload[item_key] = kept
        meta = dict(raw.get(meta_key) or {})
        meta["solvability_verification"] = verification_meta
        payload[meta_key] = meta
    else:
        payload = kept  # type: ignore[assignment]
        LOGGER.info(
            "%s is a bare JSON list, so the filtered copy is a bare list too; "
            "verification metadata lives in the verdict file only",
            source_path.name,
        )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def default_output_paths(data_path: Path, meta: Dict[str, Any]) -> Tuple[Path, Path]:
    parts = [str(meta.get(key) or "") for key in ("year", "level")]
    stem = "_".join(p for p in parts if p and p != "unknown") or data_path.stem
    base = REPO_ROOT / "outputs" / "verification"
    return base / f"{stem}_verdicts.json", base / f"{stem}_verified.json"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify_solvability.py",
        description=(
            "Step 4 Contextual Solvability Verification: keep only the questions a "
            "verifier model can answer from their own gold provenance, with no retrieval."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/verify_solvability.py demo.json --dry-run --resolve-sparql\n"
            "  python scripts/verify_solvability.py bench/2025/level2.json --n-votes 3 --concurrency 8\n"
        ),
    )
    parser.add_argument("data", help="benchmark split: demo.json or bench/<year>/level<N>.json")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="verifier model name (default: %(default)s)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible base URL; overrides OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=None,
                        help="API key; overrides OPENAI_API_KEY / .env")
    parser.add_argument("--n-votes", type=int, default=1, metavar="N",
                        help="samples per instance; kept on a strict majority of exact matches "
                             "(default: %(default)s, use an odd number)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature (default: 0.0 for one vote, 0.7 when --n-votes > 1)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="max tokens per verifier reply (default: %(default)s)")
    parser.add_argument("--max-attempts", type=int, default=4,
                        help="bounded retries per API call (default: %(default)s)")
    parser.add_argument("--concurrency", type=int, default=4, metavar="N",
                        help="worker threads (default: %(default)s)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="verify only the first N instances")
    parser.add_argument("--verdicts", default=None,
                        help="per-instance verdict file "
                             "(default: outputs/verification/<year>_<level>_verdicts.json)")
    parser.add_argument("--filtered", default=None,
                        help="filtered split holding only the instances that passed "
                             "(default: outputs/verification/<year>_<level>_verified.json)")
    parser.add_argument("--resume", action="store_true",
                        help="reuse decided verdicts from an existing --verdicts file "
                             "and re-run only what is missing or errored")
    parser.add_argument("--on-missing-context", choices=("reject", "keep", "skip"), default="reject",
                        help="what to do with an instance that carries no usable provenance: "
                             "reject it, keep it unverified, or set it aside as unverifiable "
                             "(default: %(default)s)")
    parser.add_argument("--resolve-sparql", action="store_true",
                        help="for instances with no provenance, rebuild a weak gold context by "
                             "running sparql_verification through WDQS and labelling the result "
                             "(needs network; reflects today's Wikidata)")
    parser.add_argument("--dry-run", action="store_true",
                        help="use the deterministic stub verifier: no API key, no network, "
                             "and a few instances deliberately fail so the reject path runs")
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="bootstrap confidence level (default: %(default)s)")
    parser.add_argument("--resamples", type=int, default=10000,
                        help="bootstrap resamples (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed (default: %(default)s)")
    parser.add_argument("--allow-errors", action="store_true",
                        help="exit 0 even when some instances failed with a verifier error")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="logging verbosity (default: %(default)s)")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.n_votes < 1:
        raise SystemExit("--n-votes must be at least 1")
    if args.n_votes % 2 == 0:
        LOGGER.warning("--n-votes %d is even; a tied vote is a rejection", args.n_votes)
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")

    data_path = Path(args.data)
    try:
        items, meta = dataio.load_instances(data_path)
    except dataio.DatasetFormatError as exc:
        raise SystemExit(str(exc))
    if args.limit and args.limit < len(items):
        LOGGER.warning("--limit %d: only the first %d of %d instances are verified, and the "
                       "filtered split will hold no more than that", args.limit, args.limit, len(items))
        items = items[: args.limit]
    LOGGER.info("loaded %d instance(s) from %s", len(items), data_path)

    default_verdicts, default_filtered = default_output_paths(data_path, meta)
    verdicts_path = dataio.ensure_parent(Path(args.verdicts) if args.verdicts else default_verdicts)
    filtered_path = dataio.ensure_parent(Path(args.filtered) if args.filtered else default_filtered)
    if verdicts_path.resolve() == filtered_path.resolve():
        raise SystemExit("--verdicts and --filtered must be different files")

    temperature = args.temperature
    if temperature is None:
        temperature = 0.0 if args.n_votes == 1 else 0.7

    if args.dry_run:
        verifier: Any = StubVerifier()
        model_name = f"stub (dry-run, would call {args.model})"
        LOGGER.info("dry run: using the deterministic stub verifier, no API calls will be made")
    else:
        try:
            base_url, api_key = config.openai_credentials(base_url=args.base_url, api_key=args.api_key)
        except config.MissingCredential as exc:
            raise SystemExit(f"{exc}\n  Or run with --dry-run to exercise the pipeline offline.")
        verifier = OpenAIVerifier(
            model=args.model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=args.max_tokens,
            max_attempts=args.max_attempts,
        )
        model_name = args.model
        LOGGER.info("verifier %s at %s, %d vote(s), temperature %.2f",
                    args.model, base_url, args.n_votes, temperature)

    resolver = SparqlContextResolver() if args.resolve_sparql else None

    cached: Dict[str, Dict[str, Any]] = {}
    if args.resume and verdicts_path.is_file():
        try:
            previous = json.loads(verdicts_path.read_text(encoding="utf-8"))
            for record in previous.get("verdicts", []):
                if record.get("status") in _DECIDED or record.get("status") == "unverifiable":
                    cached[str(record.get("key"))] = record
            LOGGER.info("resume: reusing %d decided verdict(s) from %s", len(cached), verdicts_path)
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise SystemExit(f"--resume could not read {verdicts_path}: {exc}")
    elif args.resume:
        LOGGER.info("resume: no existing verdict file at %s, starting fresh", verdicts_path)

    todo = [(i, item) for i, item in enumerate(items) if instance_key(i, item) not in cached]
    LOGGER.info("verifying %d instance(s), %d reused", len(todo), len(items) - len(todo))

    done = 0
    progress_lock = threading.Lock()
    total = len(todo)

    def _run(pair: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        nonlocal done
        record = verify_instance(
            pair[0], pair[1],
            verifier=verifier,
            n_votes=args.n_votes,
            on_missing=args.on_missing_context,
            resolver=resolver,
        )
        with progress_lock:
            done += 1
            if done % 25 == 0 or done == total:
                LOGGER.info("progress %d/%d", done, total)
        return record

    try:
        if todo:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                fresh = list(pool.map(_run, todo))
        else:
            fresh = []
    finally:
        if resolver is not None:
            LOGGER.info("WDQS context resolution: %d attempted, %d recovered",
                        resolver.attempted, resolver.recovered)
            resolver.close()

    by_key = {record["key"]: record for record in fresh}
    verdicts = [cached.get(instance_key(i, item)) or by_key[instance_key(i, item)]
                for i, item in enumerate(items)]

    summary = summarise(verdicts, confidence=args.confidence,
                        resamples=args.resamples, seed=args.seed)
    verification_meta = {
        "stage": "contextual_solvability_verification",
        "verifier_model": model_name,
        "n_votes": args.n_votes,
        "temperature": temperature,
        "on_missing_context": args.on_missing_context,
        "resolve_sparql": bool(args.resolve_sparql),
        "dry_run": bool(args.dry_run),
        "source_file": str(data_path),
        "n_input": len(items),
        "n_kept": sum(1 for v in verdicts if v.get("kept")),
        "pass_rate_verified": summary["overall"]["pass_rate"],
        "retention_rate": summary["retention"]["pass_rate"],
        "n_no_context": summary["n_no_context"],
        "verdicts_file": str(verdicts_path),
    }

    verdicts_path.write_text(
        json.dumps({"metadata": {**meta, **verification_meta},
                    "summary": summary,
                    "verdicts": verdicts},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_filtered(
        source_path=data_path,
        items=items,
        kept_indices=[i for i, v in enumerate(verdicts) if v.get("kept")],
        out_path=filtered_path,
        verification_meta=verification_meta,
    )

    print(render_summary(summary, data_path=str(data_path), model=model_name))
    print(f"verdicts        -> {verdicts_path}")
    print(f"verified split  -> {filtered_path} "
          f"({verification_meta['n_kept']}/{len(items)} instances)")

    n_errors = summary["status_counts"].get("error", 0)
    if n_errors and not args.allow_errors:
        LOGGER.error("%d instance(s) ended in a verifier error and were neither kept nor rejected; "
                     "re-run with --resume, or pass --allow-errors to ignore", n_errors)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
