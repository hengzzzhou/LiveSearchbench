#!/usr/bin/env python3
"""Regenerate ``data_description.json`` from the benchmark files themselves.

The dataset description used to be hand-maintained, which let it drift out of
step with what the repository actually contains. This script derives every
number from the files on disk, so running it is the only supported way to
update the description.

Usage:
    python scripts/analysis/dataset_stats.py                 # print a report
    python scripts/analysis/dataset_stats.py --write         # update data_description.json
    python scripts/analysis/dataset_stats.py --check         # exit 1 if stale (for CI)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from livesearchbench import dataio  # noqa: E402

DEFAULT_SPLITS = sorted(str(p) for p in Path("bench").glob("*/level*.json"))
DESCRIPTION_PATH = Path("data_description.json")
TOP_N = 50


def _relation_of(item: Dict) -> str:
    """Best-effort relation label for an instance, across the three levels."""
    triple = item.get("source_triple")
    if isinstance(triple, dict):
        for key in ("property_label", "relation", "predicate_label", "property"):
            if triple.get(key):
                return str(triple[key])
    if isinstance(triple, (list, tuple)) and len(triple) >= 2:
        return str(triple[1])
    info = item.get("constraint_info")
    if isinstance(info, dict):
        for key in ("relation", "property_label", "predicate"):
            if info.get(key):
                return str(info[key])
    return ""


def _subject_of(item: Dict) -> str:
    triple = item.get("source_triple")
    if isinstance(triple, dict):
        for key in ("entity_label", "subject", "subject_label", "entity"):
            if triple.get(key):
                return str(triple[key])
    if isinstance(triple, (list, tuple)) and triple:
        return str(triple[0])
    return ""


def collect(splits: List[str]) -> Dict:
    per_split: Dict[str, Dict] = {}
    levels: Counter = Counter()
    subjects: Counter = Counter()
    relations: Counter = Counter()
    objects: Counter = Counter()
    total = 0
    verified = 0

    for path in splits:
        items, meta = dataio.load_instances(path)
        total += len(items)
        split_levels: Counter = Counter()
        for item in items:
            level = item.get("level")
            key = f"L{level}" if level is not None else "unknown"
            levels[key] += 1
            split_levels[key] += 1
            if item.get("sparql_verification"):
                verified += 1
            subject = _subject_of(item)
            if subject:
                subjects[subject] += 1
            relation = _relation_of(item)
            if relation:
                relations[relation] += 1
            answer = item.get("answer")
            if isinstance(answer, str) and answer.strip():
                objects[answer.strip()] += 1
        per_split[path] = {
            "n": len(items),
            "level_distribution": dict(sorted(split_levels.items())),
            "year": meta.get("year"),
            "source_file": meta.get("source_file"),
        }

    def _summary(counter: Counter) -> Dict:
        return {
            "total_count": len(counter),
            f"top_{TOP_N}": dict(counter.most_common(TOP_N)),
        }

    return {
        "_generated_by": "scripts/analysis/dataset_stats.py",
        "_note": (
            "Every number here is derived from the files listed under 'splits'. "
            "Do not edit by hand; run the script instead."
        ),
        "total_qa_pairs": total,
        "level_distribution": dict(sorted(levels.items())),
        "instances_with_sparql_verification": verified,
        "splits": per_split,
        "subjects": _summary(subjects),
        "relations": _summary(relations),
        "objects": _summary(objects),
    }


def render(stats: Dict) -> str:
    lines = [
        f"Total QA pairs: {stats['total_qa_pairs']}",
        f"Level distribution: {stats['level_distribution']}",
        f"With SPARQL verification: {stats['instances_with_sparql_verification']}",
        "",
        f"{'Split':30s} {'n':>5s}  levels",
    ]
    for path, info in stats["splits"].items():
        lines.append(f"{path:30s} {info['n']:5d}  {info['level_distribution']}")
    lines += [
        "",
        f"Distinct subjects:  {stats['subjects']['total_count']}",
        f"Distinct relations: {stats['relations']['total_count']}",
        f"Distinct answers:   {stats['objects']['total_count']}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("splits", nargs="*", default=None,
                        help="benchmark files (default: bench/*/level*.json)")
    parser.add_argument("--write", action="store_true",
                        help=f"overwrite {DESCRIPTION_PATH} with the computed statistics")
    parser.add_argument("--check", action="store_true",
                        help=f"exit non-zero if {DESCRIPTION_PATH} disagrees with the data")
    parser.add_argument("--output", default=str(DESCRIPTION_PATH), help="destination for --write")
    args = parser.parse_args()

    splits = args.splits or DEFAULT_SPLITS
    if not splits:
        print("No benchmark files found under bench/. Nothing to do.", file=sys.stderr)
        return 1

    stats = collect(splits)
    print(render(stats))

    if args.check:
        target = Path(args.output)
        if not target.is_file():
            print(f"\n{target} does not exist.", file=sys.stderr)
            return 1
        current = json.loads(target.read_text(encoding="utf-8"))
        claimed = current.get("total_qa_pairs")
        if claimed != stats["total_qa_pairs"]:
            print(
                f"\nSTALE: {target} claims {claimed} QA pairs but the files contain "
                f"{stats['total_qa_pairs']}.\nRun: python {Path(__file__).as_posix()} --write",
                file=sys.stderr,
            )
            return 1
        print(f"\n{target} is consistent with the data ({claimed} QA pairs).")
        return 0

    if args.write:
        dataio.ensure_parent(args.output)
        Path(args.output).write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
