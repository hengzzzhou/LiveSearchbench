#!/usr/bin/env python3
"""Structured Wikidata/SPARQL upper-bound diagnostic.

This diagnostic executes each released SPARQL verification query after
converting COUNT queries into SELECT queries when possible. It measures whether
the structured KG state can recover the gold answer string. It is an upper-bound
diagnostic, not a deployed entity-linking QA baseline, because the benchmark
provides the canonical SPARQL program.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


ENDPOINT = "https://query.wikidata.org/sparql"
COUNT_RE = re.compile(r"SELECT\s*\(COUNT\(\?([A-Za-z_][A-Za-z0-9_]*)\)\s+AS\s+\?count\)", re.I)


def load_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("qa_pairs"), list):
        return data["qa_pairs"]
    raise ValueError(f"Unsupported dataset format in {path}")


def select_query(count_query: str) -> str:
    match = COUNT_RE.search(count_query)
    if not match:
        return count_query
    var_name = match.group(1)
    query = COUNT_RE.sub(f"SELECT ?{var_name}", count_query)
    return query.strip()


def query_sparql(query: str, sleep: float) -> dict[str, Any]:
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "LiveSearchBench-SPARQLUpperBound/1.0",
    }
    response = requests.get(
        ENDPOINT,
        params={"query": query, "format": "json"},
        headers=headers,
        timeout=30,
    )
    time.sleep(sleep)
    response.raise_for_status()
    return response.json()


def labels_for_values(values: list[str], sleep: float) -> list[str]:
    qids = []
    literals = []
    for value in values:
        if value.startswith("http://www.wikidata.org/entity/Q"):
            qids.append(value.rsplit("/", 1)[-1])
        else:
            literals.append(value)
    labels = list(literals)
    if qids:
        query = """
SELECT ?item ?itemLabel WHERE {
  VALUES ?item { %s }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
""" % " ".join(f"wd:{qid}" for qid in qids)
        data = query_sparql(query, sleep)
        for binding in data.get("results", {}).get("bindings", []):
            label = binding.get("itemLabel", {}).get("value")
            if label:
                labels.append(label)
    return labels


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def evaluate_item(item: dict[str, Any], sleep: float) -> dict[str, Any]:
    sparql = item.get("sparql_verification") or item.get("sparql_query")
    expected = item.get("answer", "")
    if not sparql:
        return {**item, "structured_answer": [], "is_correct": False, "error": "missing_sparql"}

    query = select_query(sparql)
    try:
        data = query_sparql(query, sleep)
        values = []
        for binding in data.get("results", {}).get("bindings", []):
            for cell in binding.values():
                if "value" in cell:
                    values.append(cell["value"])
        labels = labels_for_values(values, sleep)
        expected_norm = normalize(expected)
        correct = any(expected_norm == normalize(label) or expected_norm in normalize(label) for label in labels)
        return {
            "question": item.get("question"),
            "expected_answer": expected,
            "level": item.get("level"),
            "year": item.get("year"),
            "select_query": query,
            "structured_answer": labels,
            "is_correct": correct,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "question": item.get("question"),
            "expected_answer": expected,
            "level": item.get("level"),
            "year": item.get("year"),
            "select_query": query,
            "structured_answer": [],
            "is_correct": False,
            "error": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/structured_sparql_upper_bound.json"))
    parser.add_argument("--sleep", type=float, default=0.1, help="Delay between Wikidata requests")
    args = parser.parse_args()

    items = load_items(args.data)
    results = [evaluate_item(item, args.sleep) for item in items]
    total = len(results)
    correct = sum(1 for row in results if row["is_correct"])
    summary = {
        "total_questions": total,
        "correct_answers": correct,
        "accuracy": correct / total if total else 0.0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("total_questions", "correct_answers", "accuracy")}, indent=2))


if __name__ == "__main__":
    main()
