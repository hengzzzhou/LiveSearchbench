#!/usr/bin/env python3
"""Compute query-topology statistics for LiveSearchBench JSON files.

The script accepts either a JSON list of QA items or a dictionary with a
``qa_pairs`` field. It only inspects released metadata and SPARQL validation
queries, so it does not require API keys or network access.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PRED_RE = re.compile(r"\bwdt:(P\d+)\b")
ENTITY_RE = re.compile(r"\bwd:(Q\d+)\b")
VAR_RE = re.compile(r"\?([A-Za-z_][A-Za-z0-9_]*)")


def load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("qa_pairs"), list):
        return data["qa_pairs"]
    raise ValueError(f"Unsupported dataset format in {path}")


def load_description(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def item_stats(item: dict[str, Any]) -> dict[str, Any]:
    sparql = item.get("sparql_verification") or item.get("sparql_query") or ""
    predicates = PRED_RE.findall(sparql)
    anchors = ENTITY_RE.findall(sparql)
    variables = sorted(set(VAR_RE.findall(sparql)))
    level = int(item.get("level", 0) or 0)
    edge_count = len(predicates)
    repeated_predicates = len(predicates) - len(set(predicates))

    if level == 1:
        family = "single-hop"
    elif level == 2:
        family = "multi-constraint intersection"
    elif level == 3:
        family = "fuzzed/indirect constrained selection"
    else:
        family = "unknown"

    signature = (
        f"L{level}|edges={edge_count}|unique_pred={len(set(predicates))}|"
        f"anchors={len(set(anchors))}|repeated_pred={repeated_predicates}"
    )
    return {
        "level": level,
        "year": item.get("year", "unknown"),
        "family": family,
        "edge_count": edge_count,
        "unique_predicates": len(set(predicates)),
        "anchor_count": len(set(anchors)),
        "variable_count": len(variables),
        "repeated_predicates": repeated_predicates,
        "signature": signature,
        "predicates": predicates,
        "anchors": anchors,
    }


def pct(numer: int, denom: int) -> str:
    return f"{100 * numer / denom:.1f}%" if denom else "0.0%"


def summarize(items: list[dict[str, Any]], description: dict[str, Any]) -> dict[str, Any]:
    rows = [item_stats(item) for item in items]
    total = len(rows)
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_level[row["level"]].append(row)

    level_summary = []
    for level in sorted(by_level):
        group = by_level[level]
        level_summary.append(
            {
                "level": level,
                "family": group[0]["family"],
                "n": len(group),
                "share": pct(len(group), total),
                "avg_edges": round(mean(r["edge_count"] for r in group), 2),
                "max_edges": max(r["edge_count"] for r in group),
                "avg_anchors": round(mean(r["anchor_count"] for r in group), 2),
                "unique_signatures": len({r["signature"] for r in group}),
            }
        )

    all_predicates = [p for row in rows for p in row["predicates"]]
    all_anchors = [a for row in rows for a in row["anchors"]]
    signatures = Counter(row["signature"] for row in rows)

    diversity = description.get("diversity_summary", {})
    desc_subjects = diversity.get("unique_subjects")
    desc_relations = diversity.get("unique_relations")
    desc_objects = diversity.get("unique_objects")
    desc_diversity_note = diversity.get("note")
    desc_total = description.get("total_qa_pairs")
    desc_level_distribution = description.get("level_distribution")

    return {
        "input_items": total,
        "description_total_qa_pairs": desc_total,
        "description_level_distribution": desc_level_distribution,
        "description_unique_subjects": desc_subjects,
        "description_unique_relations": desc_relations,
        "description_unique_objects": desc_objects,
        "description_diversity_note": desc_diversity_note,
        "demo_unique_predicates": len(set(all_predicates)),
        "demo_unique_anchors": len(set(all_anchors)),
        "demo_unique_topology_signatures": len(signatures),
        "level_summary": level_summary,
        "topology_signatures": signatures.most_common(),
    }


def markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "# LiveSearchBench Topology Statistics",
        "",
        "## Released Supplement Diversity",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| QA pairs in data description | {summary.get('description_total_qa_pairs', 'n/a')} |",
        f"| Level distribution | {format_level_distribution(summary.get('description_level_distribution'))} |",
    ]
    if summary.get("description_unique_subjects") is not None:
        lines.extend(
            [
                f"| Unique subjects | {summary.get('description_unique_subjects')} |",
                f"| Unique relations | {summary.get('description_unique_relations')} |",
                f"| Unique objects | {summary.get('description_unique_objects')} |",
            ]
        )
    else:
        lines.append("| Full-dataset diversity counts | omitted from metadata |")
    if summary.get("description_diversity_note"):
        lines.extend(["", f"Note: {summary['description_diversity_note']}"])
    lines.extend(
        [
            "",
            "## SPARQL Topology From Input JSON",
        "",
        "| Level | Family | n | Share | Avg. edges/constraints | Max edges | Avg. anchors | Unique signatures |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["level_summary"]:
        lines.append(
            "| {level} | {family} | {n} | {share} | {avg_edges} | {max_edges} | "
            "{avg_anchors} | {unique_signatures} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Topology Signatures",
            "",
            "| Signature | Count |",
            "|---|---:|",
        ]
    )
    for signature, count in summary["topology_signatures"]:
        lines.append(f"| `{signature}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def format_level_distribution(distribution: dict[str, int] | None) -> str:
    if not distribution:
        return "n/a"
    return " / ".join(f"{distribution[level]} {level}" for level in ("L1", "L2", "L3") if level in distribution)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Dataset JSON")
    parser.add_argument("--description", type=Path, help="Optional data_description.json")
    parser.add_argument("--json-out", type=Path, help="Path for machine-readable summary")
    parser.add_argument("--md-out", type=Path, help="Path for Markdown table")
    args = parser.parse_args()

    summary = summarize(load_items(args.input), load_description(args.description))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(markdown_table(summary), encoding="utf-8")
    if not args.json_out and not args.md_out:
        print(markdown_table(summary))


if __name__ == "__main__":
    main()
