"""Dataset and result-file I/O.

The released benchmark files come in three shapes, and the previous release's
runners only handled one of them:

* a bare JSON list                       -- ``demo.json``
* ``{"metadata": ..., "qa_pairs": [...]}``   -- five of the six ``bench/`` files
* ``{"dataset_info": ..., "qa_pairs": [...]}`` -- ``bench/2025/level3.json``

:func:`load_instances` accepts all of them and returns a uniform view.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

#: Keys under which a wrapper object may store its header block.
_META_KEYS = ("metadata", "dataset_info", "info")
#: Keys under which a wrapper object may store the instances.
_ITEM_KEYS = ("qa_pairs", "questions", "instances", "data")


class DatasetFormatError(ValueError):
    """Raised when a file cannot be interpreted as a benchmark split."""


def load_instances(path) -> Tuple[List[Dict], Dict]:
    """Load a benchmark split. Returns ``(instances, metadata)``.

    ``metadata`` always carries ``source_file``, and ``level``/``year`` when
    they can be inferred from the path or the instances themselves.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetFormatError(
            f"No such dataset file: {path}\n"
            f"  Released splits live in bench/<year>/level<N>.json; a 30-item\n"
            f"  sample is at demo.json. Run 'ls bench/*/*.json' to see them."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(f"{path} is not valid JSON: {exc}") from exc

    meta: Dict = {}
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        for key in _META_KEYS:
            if isinstance(raw.get(key), dict):
                meta = dict(raw[key])
                break
        items = None
        for key in _ITEM_KEYS:
            if isinstance(raw.get(key), list):
                items = raw[key]
                break
        if items is None:
            raise DatasetFormatError(
                f"{path} is a JSON object but has no instance list.\n"
                f"  Expected one of: {', '.join(_ITEM_KEYS)}\n"
                f"  Found keys: {', '.join(sorted(raw)) or '(none)'}"
            )
    else:
        raise DatasetFormatError(f"{path} must contain a list or an object, got {type(raw).__name__}")

    if not items:
        raise DatasetFormatError(f"{path} contains zero instances.")

    missing = [i for i, it in enumerate(items)
               if not isinstance(it, dict) or "question" not in it or "answer" not in it]
    if missing:
        raise DatasetFormatError(
            f"{path}: {len(missing)} instance(s) lack a 'question' or 'answer' field "
            f"(first at index {missing[0]})."
        )

    # The released files carry their own provenance under "source_file"
    # (e.g. "level1_passed_at_least_one.json"), so that key is left untouched
    # and the path we actually read is recorded separately.
    meta["dataset_path"] = str(path)
    meta.setdefault("source_file", str(path))
    meta.setdefault("n", len(items))
    inferred_level = infer_level(path, items)
    if inferred_level:
        meta.setdefault("level", inferred_level)
    inferred_year = infer_year(path, items)
    if inferred_year:
        meta.setdefault("year", inferred_year)
    return items, meta


def infer_level(path, items: Sequence[Dict]) -> Optional[str]:
    """Infer the difficulty tier from the instances, falling back to the path."""
    levels = {str(it.get("level")) for it in items if it.get("level") is not None}
    if len(levels) == 1:
        return f"level{levels.pop()}"
    match = re.search(r"level\s*([123])", Path(path).name, re.IGNORECASE)
    return f"level{match.group(1)}" if match else None


def infer_year(path, items: Sequence[Dict]) -> Optional[str]:
    """Infer the release year from the instances, falling back to the path.

    Only the directory component is consulted, so a model name or timestamp
    elsewhere in the path cannot be mistaken for the split year.
    """
    years = {str(it.get("year")) for it in items if it.get("year") is not None}
    if len(years) == 1:
        return years.pop()
    for part in reversed(Path(path).resolve().parts[:-1]):
        if re.fullmatch(r"(19|20)\d{2}", part):
            return part
    return None


def normalize_instance(item: Dict) -> Dict:
    """Return a copy with the fields the runners rely on always present."""
    out = dict(item)
    out["question"] = str(item.get("question", "")).strip()
    out["answer"] = item.get("answer", "")
    out.setdefault("level", None)
    out.setdefault("sparql_verification", "")
    out.setdefault("answer_aliases", [])
    return out


def save_run(
    *,
    results: Sequence[Dict],
    summary: Dict,
    method: str,
    model_name: str,
    data_path: str,
    output_dir: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Dict[str, str]:
    """Write per-item results and a summary; returns the two paths.

    Filenames embed the search budget when the summary reports one, so that
    ``scripts/analysis/summarize_rag_budget.py`` can recover it.
    """
    from datetime import datetime

    meta = dict(metadata or {})
    level = meta.get("level") or infer_level(data_path, results) or "unknown"
    year = meta.get("year") or infer_year(data_path, []) or "unknown"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9]+", "_", str(model_name)).strip("_") or "model"

    budget = summary.get("max_search_calls_allowed")
    budget_tag = f"_maxiter_{budget}" if budget is not None else ""

    out_dir = Path(output_dir or os.path.join("outputs", "evaluations", str(year)))
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{level}_{method}_{safe_model}{budget_tag}_{stamp}"
    results_path = out_dir / f"{stem}_results.json"
    summary_path = out_dir / f"{stem}_summary.json"

    run_meta = {"method": method, "model": model_name,
                "data_file": str(data_path), "timestamp": stamp}
    payload = {"metadata": {**meta, **run_meta}, "results": list(results)}
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # ``summary`` describes THIS run and must win over any key of the same name
    # carried in the dataset's own header block. The released splits define
    # ``total_questions``, so merging the other way round silently reported the
    # split's size instead of the number of items actually evaluated.
    summary_path.write_text(json.dumps({**meta, **run_meta, **summary},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    return {"results": str(results_path), "summary": str(summary_path)}


def load_results(path) -> Tuple[List[Dict], Dict]:
    """Load a results file written by :func:`save_run` or by the old runners."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw, {"source_file": str(path)}
    if isinstance(raw, dict):
        for key in ("results", "items", "qa_pairs"):
            if isinstance(raw.get(key), list):
                meta = dict(raw.get("metadata") or {})
                meta.setdefault("source_file", str(path))
                return raw[key], meta
    raise DatasetFormatError(f"{path} does not look like a results file.")


def ensure_parent(path) -> Path:
    """Create the parent directory of ``path`` and return the path.

    Called before long-running jobs start so an unwritable destination fails
    immediately rather than after the work is done.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
