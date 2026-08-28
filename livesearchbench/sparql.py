"""Wikidata Query Service client used for uniqueness validation.

The endpoint is configurable (``--endpoint`` / ``SPARQL_ENDPOINT``) so the
verification programs can be replayed against a dated mirror rather than only
against live Wikidata, and failures are reported as failures instead of being
silently folded into "no results".
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from . import config
from .http import PoliteSession, RequestFailed

logger = logging.getLogger("livesearchbench.sparql")

_COUNT_SELECT = re.compile(r"SELECT\s*\(\s*COUNT\s*\(\s*(\??\w+)\s*\)\s+AS\s+\?(\w+)\s*\)", re.IGNORECASE)


class SparqlError(RuntimeError):
    """Raised when a SPARQL query cannot be executed."""


class SparqlClient:
    """Minimal WDQS client with bounded retries."""

    def __init__(self, *, endpoint: Optional[str] = None, session: Optional[PoliteSession] = None,
                 min_interval: float = 1.0) -> None:
        self.endpoint = config.get("SPARQL_ENDPOINT", override=endpoint,
                                   default=config.DEFAULT_SPARQL_ENDPOINT)
        self.session = session or PoliteSession(component="LiveSearchBench-SPARQL",
                                                min_interval=min_interval)

    def query(self, sparql: str) -> Dict:
        """Execute a query and return the parsed JSON results."""
        try:
            response = self.session.get(
                self.endpoint,
                params={"query": sparql, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
            )
        except RequestFailed as exc:
            raise SparqlError(str(exc)) from exc
        if response.status_code != 200:
            raise SparqlError(f"HTTP {response.status_code} from {self.endpoint}: {response.text[:200]}")
        try:
            return response.json()
        except ValueError as exc:
            raise SparqlError(f"Non-JSON response from {self.endpoint}: {response.text[:200]}") from exc

    def count(self, sparql: str) -> int:
        """Run a COUNT query and return the integer.

        Raises :class:`SparqlError` on failure rather than returning 0, so a
        network problem is never mistaken for "this query matched nothing".
        """
        data = self.query(sparql)
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return 0
        row = bindings[0]
        for key in ("count", "c", "cnt"):
            if key in row:
                return int(row[key]["value"])
        for value in row.values():
            try:
                return int(value["value"])
            except (KeyError, TypeError, ValueError):
                continue
        raise SparqlError(f"COUNT query returned no numeric binding: {row}")

    def is_unique(self, sparql: str) -> bool:
        """True when the verification program yields exactly one answer."""
        return self.count(sparql) == 1

    def select_labels(self, sparql: str, *, lang: str = "en") -> List[str]:
        """Run a SELECT and return human-readable labels for the bindings."""
        data = self.query(sparql)
        out: List[str] = []
        for row in data.get("results", {}).get("bindings", []):
            literals, uris = [], []
            for value in row.values():
                text = value.get("value", "")
                if not text:
                    continue
                if value.get("type") == "uri":
                    uris.append(text.rsplit("/", 1)[-1])
                    continue
                if value.get("xml:lang") and value["xml:lang"] != lang:
                    continue
                literals.append(text)
            # Prefer a human-readable literal; fall back to the bare QID so a
            # caller can still tell what was returned.
            out.extend(literals or uris)
        return out


def count_to_select(sparql: str) -> Optional[str]:
    """Rewrite ``SELECT (COUNT(?x) AS ?count) WHERE {...}`` into ``SELECT ?x``.

    Returns ``None`` when the query is not of that shape.
    """
    match = _COUNT_SELECT.search(sparql or "")
    if not match:
        return None
    var = match.group(1)
    if not var.startswith("?"):
        var = "?" + var
    return _COUNT_SELECT.sub(f"SELECT DISTINCT {var}", sparql, count=1)


def with_labels(sparql: str, *, lang: str = "en") -> str:
    """Attach the Wikidata label service to a SELECT query."""
    if "wikibase:label" in sparql:
        return sparql
    idx = sparql.rfind("}")
    if idx == -1:
        return sparql
    service = (f'\n  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}\n')
    return sparql[:idx] + service + sparql[idx:]


def to_label_select(sparql: str, *, lang: str = "en") -> Optional[str]:
    """Turn a COUNT verification program into a label-returning SELECT.

    ``SELECT (COUNT(?x) AS ?count) WHERE {...}`` becomes
    ``SELECT DISTINCT ?x ?xLabel WHERE {... SERVICE wikibase:label ...}``,
    which is what the SPARQL upper-bound diagnostic needs in order to compare
    the returned entity against the gold answer string.
    """
    match = _COUNT_SELECT.search(sparql or "")
    if not match:
        return None
    var = match.group(1).lstrip("?")
    projected = f"SELECT DISTINCT ?{var} ?{var}Label"
    rewritten = _COUNT_SELECT.sub(projected, sparql, count=1)
    return with_labels(rewritten, lang=lang)
