#!/usr/bin/env python3
"""Recompute LiveSearchBench evaluation metrics from saved result files.

The evaluation runners in ``scripts/eval/`` decide correctness with a
case-folded substring test (``simple_match``) and then report the resulting
number as "Exact Match". That is not exact match: a verbose answer that merely
mentions the gold string is counted correct. This script re-scores the saved
predictions from scratch with :mod:`livesearchbench.scoring` and prints
normalised exact match, token F1 and the legacy containment metric side by
side, each with a bootstrap confidence interval, so the gap between them is
visible rather than hidden behind a label.

Inputs are result files written by :func:`livesearchbench.dataio.save_run`
(``{"metadata": ..., "results": [...]}``) or by the older runners (a bare JSON
list). Globs are accepted and expanded internally, so quoting them is safe.

No API keys and no network access are required.

Examples::

    python scripts/analysis/score.py outputs/evaluations/2025/*_results.json
    python scripts/analysis/score.py run.json --format markdown -o table.md
    python scripts/analysis/score.py samples.json --pass-at-k 1 --pass-at-k 4
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from livesearchbench import dataio, scoring  # noqa: E402

LOGGER = logging.getLogger("score")

#: Metrics recomputed from the stored prediction text, in report order.
RECOMPUTED_METRICS = ("exact_match", "token_f1", "contains_match")
#: Metric read straight off the runner's own ``is_correct`` flag.
RECORDED_METRIC = "recorded"
ALL_METRICS = RECOMPUTED_METRICS + (RECORDED_METRIC,)

METRIC_LABELS = {
    "exact_match": "EM",
    "token_f1": "F1",
    "contains_match": "Contains",
    RECORDED_METRIC: "Recorded",
}

#: Result-record keys that may hold the model's answer, in preference order.
PREDICTION_KEYS = ("model_answer", "prediction", "predicted_answer", "answer_pred", "output")
#: Result-record keys that may hold the gold answer, in preference order.
GOLD_KEYS = ("expected_answer", "gold_answer", "gold", "answer", "reference_answer")
#: Result-record keys that may hold acceptable alternative gold surface forms.
ALIAS_KEYS = ("answer_aliases", "aliases", "gold_aliases")


class ScoringInputError(ValueError):
    """Raised when a results file cannot be scored."""


_LEVEL_RE = re.compile(r"^(?:level[\s_-]*)?([123])$", re.IGNORECASE)


def canonical_group(value: Any, group_key: Optional[str]) -> str:
    """Render a group value.

    Levels arrive both as bare integers (from a per-record ``level`` field) and
    as ``"level2"`` (from file metadata). Without this they would appear as two
    separate rows once several files are pooled.
    """
    text = str(value).strip()
    if group_key == "level":
        match = _LEVEL_RE.match(text)
        if match:
            return f"level{match.group(1)}"
    return text or "all"


# --------------------------------------------------------------------------
# Loading and normalisation
# --------------------------------------------------------------------------

def expand_inputs(patterns: Sequence[str]) -> List[Path]:
    """Expand globs and directories into a sorted, de-duplicated file list.

    A pattern that matches nothing is an error: silently scoring fewer files
    than the user asked for would understate or overstate a run.
    """
    paths: List[Path] = []
    seen = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            matches = sorted(candidate.glob("*results*.json")) or sorted(candidate.glob("*.json"))
        elif candidate.exists():
            matches = [candidate]
        else:
            matches = [Path(p) for p in sorted(glob.glob(pattern))]
        if not matches:
            raise ScoringInputError(
                f"No results file matched {pattern!r}.\n"
                f"  Runs are written to outputs/evaluations/<year>/*_results.json by\n"
                f"  livesearchbench.dataio.save_run. Check the path, or pass a glob."
            )
        for match in matches:
            if not match.is_file():
                continue
            resolved = match.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(match)
    if not paths:
        raise ScoringInputError("No readable results files were found.")
    return paths


def _first_present(record: Dict[str, Any], keys: Sequence[str]) -> Tuple[Optional[str], Any]:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return key, record[key]
    return None, None


def _truthy(value: Any) -> Optional[float]:
    """Interpret a recorded correctness flag; ``None`` when it is absent."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return 1.0 if float(value) > 0 else 0.0
    text = str(value).strip().lower()
    if text in ("true", "yes", "correct", "1"):
        return 1.0
    if text in ("false", "no", "incorrect", "0"):
        return 0.0
    return None


def normalize_records(
    records: Sequence[Dict[str, Any]],
    *,
    meta: Dict[str, Any],
    source: Path,
    group_key: Optional[str],
    prediction_key: Optional[str] = None,
    gold_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Map heterogeneous result records onto the keys ``scoring`` expects.

    Returns ``(items, notes)``; ``notes`` records how the fields were resolved,
    including whether the file was missing predictions and had to fall back to
    the runner's own ``is_correct`` flag.
    """
    pred_keys = (prediction_key,) if prediction_key else PREDICTION_KEYS
    gold_lookup = (gold_key,) if gold_key else GOLD_KEYS

    fallback_level = meta.get("level") or dataio.infer_level(source, records)
    items: List[Dict[str, Any]] = []
    n_with_prediction = 0
    n_with_recorded = 0
    n_missing_gold = 0
    used_pred_keys = set()
    used_gold_keys = set()

    for record in records:
        if not isinstance(record, dict):
            raise ScoringInputError(f"{source}: result records must be objects, got {type(record).__name__}")
        pkey, prediction = _first_present(record, pred_keys)
        gkey, gold = _first_present(record, gold_lookup)
        if pkey:
            used_pred_keys.add(pkey)
            n_with_prediction += 1
        if gkey:
            used_gold_keys.add(gkey)
        else:
            n_missing_gold += 1
        _, aliases = _first_present(record, ALIAS_KEYS)
        recorded = _truthy(record.get("is_correct"))
        if recorded is not None:
            n_with_recorded += 1

        group = record.get(group_key) if group_key else None
        if group in (None, ""):
            group = fallback_level or "all"
        group = canonical_group(group, group_key)

        items.append({
            "question": record.get("question", ""),
            "model_answer": "" if prediction is None else str(prediction),
            "expected_answer": "" if gold is None else gold,
            "answer_aliases": list(aliases) if isinstance(aliases, (list, tuple)) else [],
            "level": group,
            "recorded": recorded,
            "search_count": record.get("search_count"),
        })

    notes = {
        "n": len(items),
        "prediction_field": sorted(used_pred_keys) or None,
        "gold_field": sorted(used_gold_keys) or None,
        "n_with_prediction": n_with_prediction,
        "n_with_recorded_flag": n_with_recorded,
        "n_missing_gold": n_missing_gold,
        "used_is_correct_fallback": n_with_prediction == 0 and n_with_recorded > 0,
    }

    if n_with_prediction == 0 and n_with_recorded == 0:
        raise ScoringInputError(
            f"{source}: no record carries a prediction ({', '.join(pred_keys)}) or an\n"
            f"  'is_correct' flag, so there is nothing to score. Keys on the first\n"
            f"  record: {', '.join(sorted(records[0])) if records else '(none)'}"
        )
    if n_with_prediction and n_missing_gold == len(items):
        raise ScoringInputError(
            f"{source}: predictions are present but no gold answer field was found\n"
            f"  (looked for {', '.join(gold_lookup)}). Pass --gold-key to name it."
        )
    return items, notes


def load_scored_file(
    path: Path,
    *,
    group_key: Optional[str],
    prediction_key: Optional[str],
    gold_key: Optional[str],
) -> Dict[str, Any]:
    """Load one results file and normalise it. Does not compute metrics yet."""
    try:
        records, meta = dataio.load_results(path)
    except dataio.DatasetFormatError as exc:
        raise ScoringInputError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise ScoringInputError(f"{path} is not valid JSON: {exc}") from exc
    if not records:
        raise ScoringInputError(f"{path} contains zero result records.")
    items, notes = normalize_records(
        records, meta=meta, source=path, group_key=group_key,
        prediction_key=prediction_key, gold_key=gold_key,
    )
    return {"path": path, "meta": meta, "items": items, "notes": notes}


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------

def _ci(values: Sequence[float], *, confidence: float, resamples: int, seed: int) -> Dict[str, float]:
    stats = scoring.bootstrap_ci(values, confidence=confidence, resamples=resamples, seed=seed)
    return {
        "value": round(stats["mean"], 2),
        "ci_low": round(stats["lo"], 2),
        "ci_high": round(stats["hi"], 2),
    }


def _recorded_block(
    items: Sequence[Dict[str, Any]],
    *,
    group_key: Optional[str],
    confidence: float,
    resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap the runner's own ``is_correct`` flag, overall and per group."""
    flagged = [it for it in items if it.get("recorded") is not None]
    if not flagged:
        return {}

    def summarise(subset: Sequence[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        vals = [100.0 * float(it["recorded"]) for it in subset]
        return _ci(vals, confidence=confidence, resamples=resamples, seed=seed) if vals else None

    block: Dict[str, Any] = {"overall": summarise(flagged)}
    if group_key:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for it in flagged:
            groups.setdefault(str(it.get(group_key, "all")), []).append(it)
        block["groups"] = {name: summarise(recs) for name, recs in sorted(groups.items())}
    return block


def score_items(
    items: Sequence[Dict[str, Any]],
    *,
    notes: Dict[str, Any],
    group_key: Optional[str],
    confidence: float,
    resamples: int,
    seed: int,
    pass_at_ks: Sequence[int] = (),
) -> Dict[str, Any]:
    """Compute the full metric block for one already-normalised item list."""
    fallback = bool(notes.get("used_is_correct_fallback"))

    if fallback:
        overall: Dict[str, Any] = {"n": len(items)}
        by_group: Dict[str, Any] = {}
        for metric in RECOMPUTED_METRICS:
            overall[metric] = None
    else:
        agg = scoring.aggregate(
            items,
            prediction_key="model_answer",
            gold_key="expected_answer",
            alias_key="answer_aliases",
            group_key=group_key,
            confidence=confidence,
            resamples=resamples,
            seed=seed,
        )
        overall = dict(agg["overall"])
        by_group = dict(agg.get(f"by_{group_key}", {})) if group_key else {}

    recorded = _recorded_block(
        items, group_key=group_key, confidence=confidence, resamples=resamples, seed=seed,
    )
    if recorded:
        overall[RECORDED_METRIC] = recorded["overall"]
        for name, block in recorded.get("groups", {}).items():
            row = by_group.setdefault(name, {"n": 0})
            row[RECORDED_METRIC] = block
            if fallback:
                row["n"] = sum(1 for it in items if str(it.get(group_key)) == name)
                for metric in RECOMPUTED_METRICS:
                    row.setdefault(metric, None)
    else:
        overall[RECORDED_METRIC] = None

    out: Dict[str, Any] = {
        "n": len(items),
        "overall": overall,
        "by_group": by_group,
        "notes": dict(notes),
    }
    if pass_at_ks:
        out["pass_at_k"] = compute_pass_at_k(items, pass_at_ks, fallback=fallback)
    return out


def compute_pass_at_k(
    items: Sequence[Dict[str, Any]],
    ks: Sequence[int],
    *,
    fallback: bool = False,
) -> Dict[str, Any]:
    """Group samples by question and estimate pass@k for the binary metrics.

    Questions with fewer than ``k`` samples are excluded from that ``k`` and
    counted in ``skipped``, so a single-sample file does not silently inflate
    pass@4 to plain accuracy.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(scoring.normalize_answer(item.get("question", "")), []).append(item)

    metrics = [RECORDED_METRIC] if fallback else ["exact_match", "contains_match", RECORDED_METRIC]
    sample_counts = sorted(len(v) for v in groups.values())
    out: Dict[str, Any] = {
        "n_questions": len(groups),
        "samples_per_question_min": sample_counts[0] if sample_counts else 0,
        "samples_per_question_max": sample_counts[-1] if sample_counts else 0,
        "values": {},
    }

    for metric in metrics:
        per_k: Dict[str, Any] = {}
        for k in ks:
            estimates: List[float] = []
            skipped = 0
            for samples in groups.values():
                n_samples = len(samples)
                if n_samples < k:
                    skipped += 1
                    continue
                if metric == RECORDED_METRIC:
                    flags = [s.get("recorded") for s in samples]
                    if any(f is None for f in flags):
                        skipped += 1
                        continue
                    n_correct = sum(int(f) for f in flags)
                else:
                    fn = scoring.exact_match if metric == "exact_match" else scoring.contains_match
                    n_correct = sum(
                        1 for s in samples
                        if fn(s["model_answer"], s["expected_answer"], s["answer_aliases"])
                    )
                estimates.append(scoring.pass_at_k(n_samples, n_correct, k))
            per_k[str(k)] = {
                "value": round(100.0 * sum(estimates) / len(estimates), 2) if estimates else None,
                "n_questions": len(estimates),
                "skipped": skipped,
            }
        out["values"][metric] = per_k
    return out


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _fmt_cell(block: Optional[Dict[str, float]], *, show_ci: bool) -> str:
    if not block or block.get("value") is None:
        return "n/a"
    if not show_ci:
        return f"{block['value']:.2f}"
    return f"{block['value']:.2f} [{block['ci_low']:.2f}, {block['ci_high']:.2f}]"


def _rows_for(report: Dict[str, Any]) -> List[Tuple[str, str, Dict[str, Any]]]:
    """Flatten a report into ``(scope, group, metric_block)`` display rows."""
    rows: List[Tuple[str, str, Dict[str, Any]]] = []
    for entry in report["files"] + ([report["combined"]] if report.get("combined") else []):
        label = entry["label"]
        rows.append((label, "overall", entry["scores"]["overall"]))
        for name, block in sorted(entry["scores"].get("by_group", {}).items()):
            rows.append((label, name, block))
    return rows


def render_table(report: Dict[str, Any], *, show_ci: bool) -> str:
    header = ["File", "Group", "n"] + [METRIC_LABELS[m] for m in ALL_METRICS]
    body: List[List[str]] = []
    for label, group, block in _rows_for(report):
        body.append(
            [label, group, str(block.get("n", 0))]
            + [_fmt_cell(block.get(m), show_ci=show_ci) for m in ALL_METRICS]
        )

    widths = [len(h) for h in header]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells: Sequence[str]) -> str:
        parts = [cells[0].ljust(widths[0]), cells[1].ljust(widths[1])]
        parts += [cells[i].rjust(widths[i]) for i in range(2, len(cells))]
        return "  ".join(parts).rstrip()

    out = [
        "LiveSearchBench re-scored results",
        f"metrics on a 0-100 scale; brackets are {int(round(report['confidence'] * 100))}% "
        f"bootstrap CIs ({report['resamples']} resamples, seed {report['seed']})"
        if show_ci else "metrics on a 0-100 scale",
        "",
        line(header),
        "-" * len(line(header)),
    ]
    out += [line(row) for row in body]
    out.append("")
    out.append("EM = normalised exact match, F1 = token F1, Contains = legacy substring")
    out.append("containment ('simple_match'), Recorded = the is_correct flag as stored by the runner.")
    out += _footnotes(report)
    out += _pass_at_k_lines(report)
    return "\n".join(out).rstrip() + "\n"


def render_markdown(report: Dict[str, Any], *, show_ci: bool) -> str:
    conf = int(round(report["confidence"] * 100))
    lines = [
        "# LiveSearchBench Re-scored Results",
        "",
        f"All figures on a 0-100 scale. Bracketed ranges are {conf}% percentile bootstrap "
        f"confidence intervals ({report['resamples']} resamples, seed {report['seed']}).",
        "",
        "| File | Group | n | EM | F1 | Contains | Recorded |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, group, block in _rows_for(report):
        cells = [_fmt_cell(block.get(m), show_ci=show_ci) for m in ALL_METRICS]
        lines.append(
            f"| `{label}` | {group} | {block.get('n', 0)} | " + " | ".join(cells) + " |"
        )
    lines += [
        "",
        "**EM** is normalised exact match; **F1** is SQuAD-style token F1; **Contains** is the "
        "case-folded substring containment used by the original runners under the name "
        "`simple_match`; **Recorded** is the `is_correct` flag exactly as the runner stored it. "
        "Contains and Recorded are upper bounds on EM, which is why they are reported separately.",
    ]
    footnotes = _footnotes(report)
    if footnotes:
        lines += [""] + [f"- {note.strip()}" for note in footnotes if note.strip()]
    pk = _pass_at_k_markdown(report)
    if pk:
        lines += [""] + pk
    return "\n".join(lines).rstrip() + "\n"


def _footnotes(report: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    for entry in report["files"]:
        n = entry["scores"]["notes"]
        if n.get("used_is_correct_fallback"):
            notes.append(
                f"{entry['label']}: no prediction text stored, so EM/F1/Contains could not be "
                f"recomputed; only the recorded is_correct flag is reported."
            )
        elif n.get("n_with_prediction", 0) < n.get("n", 0):
            notes.append(
                f"{entry['label']}: {n['n'] - n['n_with_prediction']} of {n['n']} records have an "
                f"empty prediction; they are scored as incorrect."
            )
        if n.get("n_missing_gold"):
            notes.append(
                f"{entry['label']}: {n['n_missing_gold']} of {n['n']} records have no gold answer."
            )
    combined = report.get("combined")
    if combined and combined["scores"]["notes"].get("excluded_files"):
        excluded = combined["scores"]["notes"]["excluded_files"]
        notes.append(
            f"{combined['label']}: excludes {len(excluded)} file(s) without stored predictions "
            f"({', '.join(excluded)})."
        )
    return ([""] + notes) if notes else []


def _pass_at_k_lines(report: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for entry in report["files"] + ([report["combined"]] if report.get("combined") else []):
        pk = entry["scores"].get("pass_at_k")
        if not pk:
            continue
        lines += [
            "",
            f"pass@k for {entry['label']}: {pk['n_questions']} distinct questions, "
            f"{pk['samples_per_question_min']}-{pk['samples_per_question_max']} samples each",
        ]
        for metric, per_k in pk["values"].items():
            parts = []
            for k, block in per_k.items():
                value = "n/a" if block["value"] is None else f"{block['value']:.2f}"
                suffix = f" (skipped {block['skipped']})" if block["skipped"] else ""
                parts.append(f"pass@{k}={value}{suffix}")
            lines.append(f"  {METRIC_LABELS[metric]:<9} " + "  ".join(parts))
    return lines


def _pass_at_k_markdown(report: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for entry in report["files"] + ([report["combined"]] if report.get("combined") else []):
        pk = entry["scores"].get("pass_at_k")
        if not pk:
            continue
        ks = sorted({k for per_k in pk["values"].values() for k in per_k}, key=int)
        if not lines:
            lines.append("## pass@k")
        lines += [
            "",
            f"`{entry['label']}` -- {pk['n_questions']} distinct questions, "
            f"{pk['samples_per_question_min']}-{pk['samples_per_question_max']} samples each.",
            "",
            "| Metric | " + " | ".join(f"pass@{k}" for k in ks) + " |",
            "|---|" + "---:|" * len(ks),
        ]
        for metric, per_k in pk["values"].items():
            cells = []
            for k in ks:
                block = per_k.get(k)
                cells.append("n/a" if not block or block["value"] is None else f"{block['value']:.2f}")
            lines.append(f"| {METRIC_LABELS[metric]} | " + " | ".join(cells) + " |")
    return lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_report(
    loaded: Sequence[Dict[str, Any]],
    *,
    group_key: Optional[str],
    confidence: float,
    resamples: int,
    seed: int,
    pass_at_ks: Sequence[int],
    combine: bool,
    label_mode: str,
) -> Dict[str, Any]:
    labels = _labels_for([entry["path"] for entry in loaded], label_mode)
    files: List[Dict[str, Any]] = []
    for entry, label in zip(loaded, labels):
        files.append({
            "label": label,
            "path": str(entry["path"]),
            "metadata": entry["meta"],
            "scores": score_items(
                entry["items"], notes=entry["notes"], group_key=group_key,
                confidence=confidence, resamples=resamples, seed=seed, pass_at_ks=pass_at_ks,
            ),
        })

    report: Dict[str, Any] = {
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "group_key": group_key,
        "files": files,
    }

    if combine and len(loaded) > 1:
        # Files with no stored predictions cannot contribute to EM/F1/Contains.
        # Pooling their items anyway would score them as empty predictions and
        # silently drag the combined figures down, so they are held out and the
        # exclusion is reported.
        scorable = [e for e in loaded if not e["notes"]["used_is_correct_fallback"]]
        excluded = [lbl for e, lbl in zip(loaded, labels) if e["notes"]["used_is_correct_fallback"]]
        pooled = scorable or loaded
        all_items = [it for entry in pooled for it in entry["items"]]
        merged_notes = {
            "n": len(all_items),
            "n_with_prediction": sum(e["notes"]["n_with_prediction"] for e in pooled),
            "n_with_recorded_flag": sum(e["notes"]["n_with_recorded_flag"] for e in pooled),
            "n_missing_gold": sum(e["notes"]["n_missing_gold"] for e in pooled),
            "used_is_correct_fallback": not scorable,
            "excluded_files": excluded,
            "n_files": len(pooled),
        }
        report["combined"] = {
            "label": f"ALL ({len(pooled)} file{'s' if len(pooled) != 1 else ''})",
            "path": None,
            "metadata": {},
            "scores": score_items(
                all_items, notes=merged_notes, group_key=group_key,
                confidence=confidence, resamples=resamples, seed=seed, pass_at_ks=pass_at_ks,
            ),
        }
    return report


def _labels_for(paths: Sequence[Path], mode: str) -> List[str]:
    """Shortest unambiguous label per file: stem, else name, else full path."""
    if mode == "path":
        return [str(p) for p in paths]
    stems = [p.stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    names = [p.name for p in paths]
    if len(set(names)) == len(names):
        return names
    return [str(p) for p in paths]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="score.py",
        description=(
            "Recompute LiveSearchBench metrics from saved evaluation results. Reports "
            "normalised exact match, token F1 and legacy substring containment side by side "
            "with bootstrap confidence intervals, overall and per level."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/analysis/score.py outputs/evaluations/2025/*_results.json\n"
            "  python scripts/analysis/score.py run.json --format markdown -o appendix.md\n"
            "  python scripts/analysis/score.py samples.json --pass-at-k 1 --pass-at-k 4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results", nargs="*", metavar="RESULTS",
        help="Result JSON files, directories, or glob patterns. Both the dataio.save_run "
             "format and the old bare-list format are accepted.",
    )
    parser.add_argument(
        "--format", choices=("table", "markdown", "json"), default="table",
        help="Output format: aligned text table (default), paste-ready markdown, or raw JSON.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Write the report to this path instead of stdout (parent dirs are created).",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.95,
        help="Confidence level for the bootstrap intervals (default: 0.95).",
    )
    parser.add_argument(
        "--resamples", type=int, default=10000,
        help="Bootstrap resamples per metric (default: 10000). Lower it for a quick look.",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Bootstrap RNG seed; the same seed gives byte-identical output (default: 0).",
    )
    parser.add_argument(
        "--no-ci", action="store_true",
        help="Print point estimates only, without the confidence intervals.",
    )
    parser.add_argument(
        "--group-by", default="level", metavar="KEY",
        help="Result-record field to break the report down by, or 'none' to skip the "
             "breakdown. Records without the field inherit the level recorded in the file's "
             "metadata or inferred from its name (default: level).",
    )
    parser.add_argument(
        "--pass-at-k", type=int, action="append", default=None, metavar="K", dest="pass_at_k",
        help="Compute pass@K by grouping samples that share a question. Repeatable, "
             "e.g. --pass-at-k 1 --pass-at-k 4.",
    )
    parser.add_argument(
        "--prediction-key", default=None, metavar="KEY",
        help=f"Force the record field holding the model answer (tried in order: "
             f"{', '.join(PREDICTION_KEYS)}).",
    )
    parser.add_argument(
        "--gold-key", default=None, metavar="KEY",
        help=f"Force the record field holding the gold answer (tried in order: "
             f"{', '.join(GOLD_KEYS)}).",
    )
    parser.add_argument(
        "--no-combined", action="store_true",
        help="Suppress the combined row that pools every input file.",
    )
    parser.add_argument(
        "--label", choices=("stem", "path"), default="stem",
        help="Row label: the file stem (default) or the full path.",
    )
    parser.add_argument(
        "--self-test", action="store_true", dest="self_test",
        help="Score a synthetic run built in a temporary directory and check the "
             "numbers against hand-computed values. Needs no inputs, credentials or "
             "network; exits non-zero if any check fails.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging on stderr.",
    )
    args = parser.parse_args(argv)

    if not args.results and not args.self_test:
        parser.error("give at least one RESULTS file, or pass --self-test")
    if not 0.0 < args.confidence < 1.0:
        parser.error("--confidence must be strictly between 0 and 1")
    if args.resamples < 1:
        parser.error("--resamples must be >= 1")
    if args.pass_at_k:
        bad = [k for k in args.pass_at_k if k < 1]
        if bad:
            parser.error(f"--pass-at-k must be >= 1, got {bad}")
    return args


def run_self_test() -> int:
    """Score a synthetic run and check the numbers against hand-computed values.

    The fixture is deliberately built so that containment and exact match
    disagree: eight of ten answers wrap the gold string in prose, so the legacy
    metric reads 80 while true exact match reads 0.
    """
    import tempfile

    gold = ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Karen Sparck Jones",
            "Barbara Liskov", "Frances Allen", "Shafi Goldwasser", "Radia Perlman",
            "Margaret Hamilton", "Jean Bartik"]
    verbose = [
        {"question": f"Who is person {i}?", "expected_answer": g,
         "model_answer": (f"Based on the sources, the answer is {g}."
                          if i < 8 else "I could not determine this."),
         "is_correct": i < 8, "level": 1}
        for i, g in enumerate(gold)
    ]
    terse = [
        {"question": f"Who is person {i}?", "expected_answer": g,
         "model_answer": g if i < 6 else "unknown", "is_correct": i < 6, "level": 2}
        for i, g in enumerate(gold)
    ]
    # Four samples per question; exactly one of the four is right for questions 0-4.
    sampled = [
        {"question": f"Who is person {i}?", "expected_answer": g,
         "model_answer": g if (i < 5 and s == 0) else "Paris",
         "is_correct": i < 5 and s == 0, "level": 1}
        for i, g in enumerate(gold) for s in range(4)
    ]
    flags_only = [{"question": f"Who is person {i}?", "expected_answer": g,
                   "is_correct": i < 7} for i, g in enumerate(gold)]

    checks: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, bool(ok), detail))

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "level1_DA_verbose_20250101_000000_results.json").write_text(
            json.dumps({"metadata": {"level": "level1"}, "results": verbose}), encoding="utf-8")
        (root / "level2_CoT_terse_20250101_000000_results.json").write_text(
            json.dumps(terse), encoding="utf-8")  # old bare-list format
        (root / "level1_DA_sampled_20250101_000000_results.json").write_text(
            json.dumps({"metadata": {"level": "level1"}, "results": sampled}), encoding="utf-8")
        (root / "legacy_flags_only_results.json").write_text(
            json.dumps(flags_only), encoding="utf-8")

        def report_for(names: Sequence[str], **kw) -> Dict[str, Any]:
            loaded = [load_scored_file(root / n, group_key="level",
                                       prediction_key=None, gold_key=None) for n in names]
            params = {"group_key": "level", "confidence": 0.95, "resamples": 200,
                      "seed": 0, "pass_at_ks": (), "combine": True, "label_mode": "stem"}
            params.update(kw)
            return build_report(loaded, **params)

        rep = report_for(["level1_DA_verbose_20250101_000000_results.json"])
        overall = rep["files"][0]["scores"]["overall"]
        check("verbose run: exact match is 0", overall["exact_match"]["value"] == 0.0,
              f"got {overall['exact_match']['value']}")
        check("verbose run: containment is 80", overall["contains_match"]["value"] == 80.0,
              f"got {overall['contains_match']['value']}")
        check("verbose run: recorded flag is 80", overall["recorded"]["value"] == 80.0,
              f"got {overall['recorded']['value']}")
        check("verbose run: F1 is strictly between EM and containment",
              0.0 < overall["token_f1"]["value"] < 80.0,
              f"got {overall['token_f1']['value']}")

        rep = report_for(["level2_CoT_terse_20250101_000000_results.json"])
        overall = rep["files"][0]["scores"]["overall"]
        check("bare-list run loads", rep["files"][0]["scores"]["n"] == 10)
        check("terse run: all three metrics agree at 60",
              overall["exact_match"]["value"] == overall["contains_match"]["value"] == 60.0,
              f"em={overall['exact_match']['value']} contains={overall['contains_match']['value']}")

        rep = report_for(["level1_DA_verbose_20250101_000000_results.json",
                          "level2_CoT_terse_20250101_000000_results.json"])
        combined = rep["combined"]["scores"]
        check("combined row pools both files", combined["n"] == 20, f"got {combined['n']}")
        check("combined row keeps a per-level breakdown",
              set(combined["by_group"]) == {"level1", "level2"}, f"got {sorted(combined['by_group'])}")
        check("combined exact match is the pooled mean",
              combined["overall"]["exact_match"]["value"] == 30.0,
              f"got {combined['overall']['exact_match']['value']}")

        rep = report_for(["level1_DA_sampled_20250101_000000_results.json"], pass_at_ks=(1, 4, 8))
        pk = rep["files"][0]["scores"]["pass_at_k"]["values"]["exact_match"]
        check("pass@1 equals plain accuracy", pk["1"]["value"] == 12.5, f"got {pk['1']['value']}")
        check("pass@4 recovers the 5 solvable questions", pk["4"]["value"] == 50.0,
              f"got {pk['4']['value']}")
        check("pass@8 is skipped with only 4 samples", pk["8"]["value"] is None and
              pk["8"]["skipped"] == 10, f"got {pk['8']}")

        rep = report_for(["legacy_flags_only_results.json",
                          "level2_CoT_terse_20250101_000000_results.json"])
        notes = rep["files"][0]["scores"]["notes"]
        check("file without predictions falls back to is_correct",
              notes["used_is_correct_fallback"] is True)
        check("fallback file reports no exact match",
              rep["files"][0]["scores"]["overall"]["exact_match"] is None)
        check("fallback file still reports the recorded flag",
              rep["files"][0]["scores"]["overall"]["recorded"]["value"] == 70.0,
              f"got {rep['files'][0]['scores']['overall']['recorded']}")
        check("fallback file is held out of the combined row",
              rep["combined"]["scores"]["notes"]["excluded_files"] == ["legacy_flags_only_results"])

        names = ["level1_DA_verbose_20250101_000000_results.json"]
        a = json.dumps(report_for(names, seed=11), sort_keys=True)
        b = json.dumps(report_for(names, seed=11), sort_keys=True)
        c = json.dumps(report_for(names, seed=12), sort_keys=True)
        check("same seed gives identical output", a == b)
        check("different seed moves the intervals", a != c)

        rep = report_for(["level1_DA_verbose_20250101_000000_results.json",
                          "level2_CoT_terse_20250101_000000_results.json"])
        for fmt, render in (("table", render_table), ("markdown", render_markdown)):
            text = render(rep, show_ci=True)
            check(f"{fmt} renderer produces every row",
                  text.count("level1") >= 1 and text.count("level2") >= 1 and "ALL" in text)
        check("json renderer is serialisable", bool(json.dumps(rep)))

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        line = f"  {'PASS' if ok else 'FAIL'}  {name}"
        if not ok and detail:
            line += f"  ({detail})"
        print(line)
    print(f"\nscore.py self-test: {len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.self_test:
        return run_self_test()

    group_key: Optional[str] = None if args.group_by.lower() == "none" else args.group_by

    try:
        paths = expand_inputs(args.results)
        # Create the destination before any scoring work begins.
        if args.output:
            dataio.ensure_parent(args.output)
        loaded = [
            load_scored_file(
                path, group_key=group_key,
                prediction_key=args.prediction_key, gold_key=args.gold_key,
            )
            for path in paths
        ]
    except ScoringInputError as exc:
        LOGGER.error("%s", exc)
        return 2
    except OSError as exc:
        LOGGER.error("cannot read or write: %s", exc)
        return 2

    LOGGER.debug("scoring %d file(s), %d records total",
                 len(loaded), sum(len(e["items"]) for e in loaded))

    report = build_report(
        loaded,
        group_key=group_key,
        confidence=args.confidence,
        resamples=args.resamples,
        seed=args.seed,
        pass_at_ks=tuple(sorted(set(args.pass_at_k or ()))),
        combine=not args.no_combined,
        label_mode=args.label,
    )

    if args.format == "json":
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    elif args.format == "markdown":
        text = render_markdown(report, show_ci=not args.no_ci)
    else:
        text = render_table(report, show_ci=not args.no_ci)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        LOGGER.info("wrote %s", args.output)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
