#!/usr/bin/env python3
"""Summarize LiveSearchBench RAG results by search budget.

This script consumes JSON result files produced by ``scripts/eval/RAG.py``.
It does not call any model or search API. Use it after running RAG.py with
different ``--max-iter`` values.

The budget attached to each file is resolved in this order:

1. ``--budget`` on the command line (applies to every file given).
2. A budget recorded inside the file, e.g. the ``max_search_calls_allowed``
   field that :func:`livesearchbench.dataio.save_run` copies into the run
   metadata.
3. The filename, which ``save_run`` stamps as ``_maxiter_<N>_``; the older
   ``budget_<N>`` / ``max_iter_<N>`` / ``iter<N>`` spellings still work.
4. ``"unknown"``.

The ``accuracy`` column is the runner's own ``is_correct`` flag, which is
computed by substring containment rather than exact match. Use
``scripts/analysis/score.py`` for exact match, token F1 and containment
side by side.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Optional

LOGGER = logging.getLogger("summarize_rag_budget")

#: Filename spellings of a search budget. ``max[_-]?iter`` covers the
#: ``_maxiter_<N>_`` tag written by ``livesearchbench.dataio.save_run`` as well
#: as the ``max_iter``/``max-iter`` forms used by hand-named files.
BUDGET_IN_NAME = re.compile(
    r"(?:budget|max[_-]?iter(?:ation)?s?|max[_-]?search(?:es|_calls)?|iter)[_-]?(\d+)",
    re.IGNORECASE,
)

#: Keys under which a run may record its own budget.
BUDGET_FIELDS = (
    "max_search_calls_allowed",
    "max_iterations",
    "max_iter",
    "search_budget",
    "budget",
)


def budget_from_payload(payload: Any) -> Optional[int]:
    """Recover a budget recorded inside a results file, if there is one."""
    if not isinstance(payload, dict):
        return None
    blocks = [payload]
    for key in ("metadata", "summary", "config", "args"):
        block = payload.get(key)
        if isinstance(block, dict):
            blocks.append(block)
    for block in blocks:
        for field in BUDGET_FIELDS:
            value = block.get(field)
            if value is None or isinstance(value, bool):
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def budget_from_name(path: Path) -> Optional[int]:
    """Recover a budget stamped into the filename, if there is one."""
    match = BUDGET_IN_NAME.search(path.name)
    return int(match.group(1)) if match else None


def infer_budget(path: Path, explicit_budget: int | None, payload: Any = None) -> int | str:
    """Resolve the search budget for one results file.

    ``explicit_budget`` wins, then a budget recorded in the file, then the
    filename, then ``"unknown"``.
    """
    if explicit_budget is not None:
        return explicit_budget
    from_payload = budget_from_payload(payload)
    if from_payload is not None:
        return from_payload
    from_name = budget_from_name(path)
    if from_name is not None:
        return from_name
    return "unknown"


def load_payload(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def rows_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    raise ValueError(
        f"Unsupported result format in {path}: expected a JSON list or an object "
        f"with a 'results' list."
    )


def load_results(path: Path) -> list[dict[str, Any]]:
    """Backwards-compatible loader kept for callers that import this module."""
    return rows_from_payload(load_payload(path), path)


def expand_inputs(patterns: list[str]) -> list[Path]:
    """Expand globs and directories; a pattern matching nothing is an error."""
    paths: list[Path] = []
    seen = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_dir():
            matches = sorted(candidate.glob("*results*.json"))
        elif candidate.exists():
            matches = [candidate]
        else:
            matches = [Path(p) for p in sorted(glob.glob(pattern))]
        if not matches:
            raise ValueError(f"No results file matched {pattern!r}")
        for match in matches:
            resolved = match.resolve()
            if match.is_file() and resolved not in seen:
                seen.add(resolved)
                paths.append(match)
    if not paths:
        raise ValueError("No readable results files were found.")
    return paths


def summarize_file(path: Path, explicit_budget: int | None) -> dict[str, Any]:
    payload = load_payload(path)
    rows = rows_from_payload(payload, path)
    total = len(rows)
    if total == 0:
        raise ValueError(f"{path} contains zero result records.")
    correct = sum(1 for row in rows if row.get("is_correct"))
    search_counts = [int(row.get("search_count", 0) or 0) for row in rows]
    n_missing_counts = sum(1 for row in rows if row.get("search_count") is None)
    if n_missing_counts == total:
        LOGGER.warning(
            "%s: no record has a 'search_count' field; search-call columns will read 0. "
            "Only scripts/eval/RAG.py records it.", path.name,
        )
    return {
        "file": str(path),
        "budget": infer_budget(path, explicit_budget, payload),
        "n": total,
        "accuracy": round(100 * correct / total, 2),
        "avg_search_calls": round(mean(search_counts), 2) if search_counts else 0.0,
        "max_search_calls": max(search_counts) if search_counts else 0,
        "zero_search_fraction": round(100 * sum(1 for c in search_counts if c == 0) / total, 2),
        "records_without_search_count": n_missing_counts,
    }


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Search-Budget Summary",
        "",
        "| Budget | n | Accuracy | Avg. search calls | Max search calls | Zero-search fraction | Source |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {budget} | {n} | {accuracy:.2f} | {avg_search_calls:.2f} | "
            "{max_search_calls} | {zero_search_fraction:.2f} | `{file}` |".format(**row)
        )
    lines += [
        "",
        "Accuracy is the runner's recorded `is_correct` flag, which uses substring "
        "containment. Run `scripts/analysis/score.py` for exact match and token F1.",
        "",
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="summarize_rag_budget.py",
        description=(
            "Summarize RAG evaluation runs by their search budget: accuracy, average and "
            "maximum search calls, and the fraction of questions answered without searching."
        ),
        epilog=(
            "Examples:\n"
            "  python scripts/analysis/summarize_rag_budget.py outputs/evaluations/2025/*_results.json\n"
            "  python scripts/analysis/summarize_rag_budget.py run.json --budget 5 --md-out table.md\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results", nargs="+", metavar="RESULTS",
        help="RAG result JSON files, directories, or glob patterns. Both the "
             "dataio.save_run format and the old bare-list format are accepted.",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help="Search budget to attribute to every file given, overriding anything "
             "recorded in the file or stamped into its name. Use it when the run was "
             "saved before the budget was written to disk.",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Write the summary rows to this JSON path (parent dirs are created).",
    )
    parser.add_argument(
        "--md-out", type=Path, default=None,
        help="Write the markdown table to this path (parent dirs are created).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging on stderr.",
    )
    args = parser.parse_args(argv)
    if args.budget is not None and args.budget < 0:
        parser.error("--budget must be >= 0")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        paths = expand_inputs(args.results)
        # Fail on an unwritable destination before any file is parsed.
        for out in (args.json_out, args.md_out):
            if out is not None:
                out.parent.mkdir(parents=True, exist_ok=True)
        if args.budget is not None and len(paths) > 1:
            LOGGER.warning("--budget %d is applied to all %d files", args.budget, len(paths))
        rows = [summarize_file(path, args.budget) for path in paths]
    except (ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        return 2

    rows.sort(key=lambda r: (999999 if r["budget"] == "unknown" else int(r["budget"]), r["file"]))

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        LOGGER.info("wrote %s", args.json_out)
    if args.md_out:
        args.md_out.write_text(markdown(rows), encoding="utf-8")
        LOGGER.info("wrote %s", args.md_out)
    if not args.json_out and not args.md_out:
        print(markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
