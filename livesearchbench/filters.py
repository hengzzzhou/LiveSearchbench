"""Candidate filtering for the construction pipeline.

Implements the three filters described in the paper's Step 2:

(i)   a curated relation allow-list, restored from ``filtered.json`` (198
      relation labels) which was removed from the repository in commit
      ``c404540``, plus a deny-list of meta/formatting property IDs;
(ii)  entity quality and disambiguation checks;
(iii) statement validity -- rank filtering and (subject, relation) dedup.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

_DATA_DIR = Path(__file__).resolve().parent / "data"

#: Meta/formatting predicates excluded from question synthesis.
#: This is the list printed as Table 4 of the paper's appendix.
EXCLUDED_PROPERTY_IDS: Dict[str, str] = {
    "P18": "image",
    "P31": "instance of (often too basic)",
    "P279": "subclass of",
    "P373": "Commons category",
    "P443": "pronunciation audio",
    "P460": "said to be the same as",
    "P856": "official website",
    "P910": "topic's main category",
    "P973": "described at URL",
    "P1151": "topic's main Wikimedia portal",
    "P1343": "described by source",
    "P1424": "topic's main template",
    "P1559": "name in native language",
    "P1629": "Wikidata property",
    "P1630": "formatter URL",
    "P1659": "related property",
    "P1687": "Wikidata property",
    "P1696": "inverse property",
    "P1705": "native label",
    "P1793": "regular expression",
    "P1855": "Wikidata property example",
    "P1889": "different from",
    "P1921": "URI template",
    "P2302": "property constraint",
    "P2700": "protocol",
    "P2875": "property for this type",
    "P2916": "source website for the property",
    "P2959": "permanent duplicated item",
    "P3254": "property usage tracking category",
    "P3709": "unit symbol",
    "P3713": "pronunciation audio",
}

#: Wikidata classes whose members make poor question subjects.
EXCLUDED_ENTITY_CLASSES: Dict[str, str] = {
    "Q4167410": "Wikimedia disambiguation page",
    "Q13406463": "Wikimedia list article",
    "Q4167836": "Wikimedia category",
    "Q11266439": "Wikimedia template",
    "Q17362920": "Wikimedia duplicated page",
    "Q13442814": "scholarly article",
}

#: Statement ranks that are dropped outright.
EXCLUDED_RANKS = frozenset({"deprecated"})

_ALLOWLIST_CACHE: Optional[Set[str]] = None


def relation_allowlist() -> Set[str]:
    """Curated set of relation *labels* retained for question synthesis."""
    global _ALLOWLIST_CACHE
    if _ALLOWLIST_CACHE is None:
        path = _DATA_DIR / "relation_allowlist.json"
        _ALLOWLIST_CACHE = {str(x).strip().lower() for x in json.loads(path.read_text(encoding="utf-8"))}
    return _ALLOWLIST_CACHE


def is_allowed_relation(property_id: str = "", property_label: str = "", *, use_allowlist: bool = True) -> bool:
    """True when a predicate may be used to build a question.

    The deny-list is always applied. The allow-list is applied only when
    ``use_allowlist`` is set, so that regenerating with a broader relation
    set remains possible without editing the library.
    """
    if property_id and property_id.upper() in EXCLUDED_PROPERTY_IDS:
        return False
    if use_allowlist:
        return (property_label or "").strip().lower() in relation_allowlist()
    return True


def is_allowed_entity(
    *,
    label: str = "",
    sitelinks: Optional[dict] = None,
    instance_of: Optional[Iterable[str]] = None,
    require_enwiki: bool = True,
) -> Tuple[bool, str]:
    """Entity quality / disambiguation check.

    Returns ``(ok, reason)``; ``reason`` is the empty string when ok.
    """
    if not (label or "").strip():
        return False, "missing label"
    if require_enwiki and "enwiki" not in (sitelinks or {}):
        return False, "no English Wikipedia sitelink"
    lowered = label.strip().lower()
    for prefix in ("category:", "template:", "wikipedia:", "list of ", "help:", "portal:"):
        if lowered.startswith(prefix):
            return False, f"surface form starts with '{prefix}'"
    for qid in instance_of or ():
        if qid in EXCLUDED_ENTITY_CLASSES:
            return False, f"instance of {EXCLUDED_ENTITY_CLASSES[qid]}"
    return True, ""


def best_statement(statements: Sequence[dict]) -> Optional[dict]:
    """Pick the statement to use for a (subject, relation) pair.

    Drops deprecated ranks, prefers ``preferred`` over ``normal``, and returns
    ``None`` when nothing survives -- the rank filtering the paper's Step 2
    (iii) describes but the previous release did not implement.
    """
    usable = [s for s in statements if s.get("rank", "normal") not in EXCLUDED_RANKS]
    if not usable:
        return None
    for statement in usable:
        if statement.get("rank") == "preferred":
            return statement
    return usable[0]


def dedup_key(subject_id: str, property_id: str, statement_id: str = "") -> str:
    """Normalised dedup key: statement ID when available, else (s, r)."""
    if statement_id:
        return f"stmt:{statement_id}"
    return f"sr:{(subject_id or '').strip().upper()}|{(property_id or '').strip().upper()}"


class FilterStats:
    """Counter for the pipeline funnel reported in the paper's Table 5."""

    def __init__(self) -> None:
        self.stages: List[Tuple[str, int]] = []
        self.drops: Dict[str, int] = {}

    def stage(self, name: str, count: int) -> None:
        self.stages.append((name, count))

    def drop(self, reason: str, n: int = 1) -> None:
        self.drops[reason] = self.drops.get(reason, 0) + n

    def to_dict(self) -> Dict:
        rows = []
        first = self.stages[0][1] if self.stages else 0
        prev = None
        for name, count in self.stages:
            rows.append({
                "stage": name,
                "count": count,
                "survival_vs_previous": round(100.0 * count / prev, 2) if prev else 100.0,
                "survival_vs_start": round(100.0 * count / first, 4) if first else 0.0,
            })
            prev = count
        return {"funnel": rows, "drop_reasons": dict(sorted(self.drops.items(), key=lambda kv: -kv[1]))}

    def render(self) -> str:
        d = self.to_dict()
        lines = ["| Pipeline Stage | Count | Survival |", "|---|---:|---:|"]
        for row in d["funnel"]:
            lines.append(f"| {row['stage']} | {row['count']:,} | {row['survival_vs_previous']}% |")
        if d["drop_reasons"]:
            lines += ["", "| Drop reason | Count |", "|---|---:|"]
            for reason, n in d["drop_reasons"].items():
                lines.append(f"| {reason} | {n:,} |")
        return "\n".join(lines)


# --- Identifier validation ------------------------------------------------
#
# Entity and property ids read from a CSV are interpolated straight into SPARQL
# strings. Validating their shape keeps a malformed or hostile row from either
# producing a silently wrong query or injecting extra clauses into whatever
# endpoint is configured.

_QID_RE = re.compile(r"^Q[1-9][0-9]*$")
_PID_RE = re.compile(r"^P[1-9][0-9]*$")


class InvalidIdentifier(ValueError):
    """Raised when a Wikidata identifier is not well formed."""


def is_qid(value: str) -> bool:
    """True for a well-formed Wikidata entity id such as ``Q42``."""
    return bool(_QID_RE.match(str(value or "").strip()))


def is_pid(value: str) -> bool:
    """True for a well-formed Wikidata property id such as ``P31``."""
    return bool(_PID_RE.match(str(value or "").strip()))


def require_qid(value: str, *, field: str = "entity id") -> str:
    """Return the normalised QID or raise :class:`InvalidIdentifier`."""
    text = str(value or "").strip()
    if not is_qid(text):
        raise InvalidIdentifier(
            f"{field} {text!r} is not a Wikidata entity id (expected Q followed by digits)"
        )
    return text


def require_pid(value: str, *, field: str = "property id") -> str:
    """Return the normalised PID or raise :class:`InvalidIdentifier`."""
    text = str(value or "").strip()
    if not is_pid(text):
        raise InvalidIdentifier(
            f"{field} {text!r} is not a Wikidata property id (expected P followed by digits)"
        )
    return text
