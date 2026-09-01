#!/usr/bin/env python3
"""Step 1: dump-differential triple extraction (the paper's Delta).

The released ``extract_triple_changes.py`` polls the live ``recentchanges``
API, which only reaches back over a rolling ~30-day window and therefore
cannot reproduce the 2021 or 2025 batches at all. This script implements the
construction step the paper actually describes: take two full Wikidata JSON
snapshots at T0 and T1, normalise each into SRO triples, and compute

    Delta = Delta_plus  union  Delta_circle

where ``Delta_plus`` holds triples whose (subject, relation) pair is absent
from T0 (emitted with ``change_type = created``) and ``Delta_circle`` holds
pairs present in both snapshots whose object changed (``change_type =
updated``). Deletions are deliberately *not* emitted: the paper defines Delta
as the union above, and ``generate_level*.py`` keeps only rows whose
``new_value`` is a Q-item, so a deletion could never become a question.

Output is a CSV with exactly the header the live extractor writes, so
``generate_level1.py``/``level2``/``level3`` accept it unchanged::

    entity_id, entity_label, property_id, property_label, property_type,
    old_value, new_value, new_value_label, change_type, change_timestamp,
    wiki_url

Peak-memory strategy
--------------------
A ``latest-all`` dump is ~100 GB uncompressed and holds ~10^8 entities, so
nothing is ever materialised in memory:

1. Both dumps are read **line by line** from the compressed stream (one JSON
   entity per line, wrapped in ``[`` / ``]``; the trailing comma is stripped).
   Decompression is streaming, so RSS is independent of file size.
2. The T0 pass writes ``(subject, property) -> (value hash, value)`` into an
   on-disk **sqlite3** table (stdlib, ``WITHOUT ROWID``, batched
   ``executemany``). Comparison uses the 8-byte BLAKE2b hash; the raw value is
   carried alongside only so the ``old_value`` column can be filled in.
3. The T1 pass streams against that index and appends candidate delta rows to
   a second sqlite table, so the candidate set is also on disk.
4. Object and property labels are collected into further sqlite tables during
   the T1 pass and joined in at emit time, because an object entity may appear
   anywhere in the stream relative to the statement that references it.

Peak RSS is therefore O(chunk_size * workers) parsed entities plus sqlite's
page cache, i.e. a few hundred MB regardless of dump size. Peak *disk* is the
index, roughly 40-60 bytes per indexed statement.

Filtering
---------
``livesearchbench.filters`` is the single source of truth:
``is_allowed_entity`` (label / enwiki / disambiguation-class checks),
``best_statement`` (drops deprecated rank, prefers ``preferred``),
``dedup_key`` ((s, r) deduplication) and ``is_allowed_relation``. The
property-ID deny-list is applied while streaming (it needs no labels); the
198-label allow-list is applied at emit time, once property labels are known.
The funnel is reported with ``filters.FilterStats`` as JSON + markdown -- the
paper's Table 5.

Provenance
----------
A MANIFEST JSON records, for each dump: path, size, streamed sha256 (unless
``--skip-hash``), mtime, and the dump date parsed from the filename. This is
the provenance the paper's conclusion promises but the repository never
shipped.

Fixtures
--------
Real dumps are ~100 GB, so ``--make-fixture DIR`` writes two tiny synthetic
snapshots in the real dump format (``data/sample/dump_T0.json.gz`` and
``dump_T1.json.gz``). They exercise, by construction: an inserted triple
(Q7259 P176 -> Q312, Delta_plus), an updated triple (Q42 P19 Q84 -> Q90,
Delta_circle), an unchanged triple (Q42 P17 Q145), a deprecated-rank statement
(Q7259 P19), a denied predicate (Q42 P18 image), a relation outside the
allow-list (Q7259 P54), an entity with no enwiki sitelink (Q999) and a
disambiguation page (Q31000). Running the extractor over them must yield
exactly two rows. The smoke test reuses this fixture.

Examples
--------
    # regenerate the fixture and run end to end over it
    python scripts/extract_dump_delta.py --make-fixture data/sample
    python scripts/extract_dump_delta.py \
        --t0 data/sample/dump_T0.json.gz \
        --t1 data/sample/dump_T1.json.gz \
        --output outputs/extracted_triples/delta_sample.csv --offline

    # a real pair of snapshots
    python scripts/extract_dump_delta.py \
        --t0 dumps/wikidata-20210104-all.json.gz \
        --t1 dumps/wikidata-20250106-all.json.gz \
        --output outputs/extracted_triples/delta_2021_2025.csv \
        --workers 8 --progress-every 1000000
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import hashlib
import json
import logging
import re
import sqlite3
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from livesearchbench import filters  # noqa: E402

logger = logging.getLogger("dump_delta")

#: CSV header shared with ``extract_triple_changes.py``.
CSV_HEADER = [
    "entity_id", "entity_label", "property_id", "property_label", "property_type",
    "old_value", "new_value", "new_value_label", "change_type", "change_timestamp",
    "wiki_url",
]

#: Property datatypes that can become an answerable question object.
DEFAULT_DATATYPES = ("wikibase-item", "time", "quantity", "globe-coordinate")

#: Sentinel written into ``old_value`` for Delta_plus rows, matching the live
#: extractor so downstream code sees no difference.
NEW_CREATED = "NEW_CREATED"

_DUMP_DATE_RE = re.compile(r"(?<!\d)(20\d{2})-?(0[1-9]|1[0-2])-?(0[1-9]|[12]\d|3[01])(?!\d)")


# ---------------------------------------------------------------------------
# Dump streaming
# ---------------------------------------------------------------------------

def open_dump(path: Path):
    """Open a Wikidata JSON dump as a text stream, chosen by extension."""
    suffix = path.suffix.lower()
    if suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    if suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8")
    if suffix in (".json", ".ndjson", ".jsonl"):
        return path.open("rt", encoding="utf-8")
    raise ValueError(
        f"Unsupported dump extension '{path.suffix}' for {path}. "
        f"Expected one of .json, .jsonl, .ndjson, .gz, .bz2."
    )


def iter_dump_lines(path: Path, *, limit: Optional[int] = None) -> Iterator[str]:
    """Yield one JSON entity string per line from a dump.

    Skips the ``[`` / ``]`` wrapper lines and strips the trailing comma. Never
    reads more than one line into memory.
    """
    seen = 0
    with open_dump(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line in ("[", "]"):
                continue
            if line.endswith(","):
                line = line[:-1]
            if not line:
                continue
            yield line
            seen += 1
            if limit is not None and seen >= limit:
                logger.info("%s: stopping at --limit %d entities", path.name, limit)
                return


# ---------------------------------------------------------------------------
# Entity normalisation (runs inside worker processes -- keep it picklable)
# ---------------------------------------------------------------------------

def canonical_value(snak: Dict[str, Any]) -> Optional[str]:
    """Canonical string form of a mainsnak value.

    Returns ``None`` for ``novalue``/``somevalue`` snaks and for datavalues we
    cannot represent, so they never enter the index or the delta. The string
    forms match ``extract_triple_changes.py`` exactly.
    """
    if snak.get("snaktype") != "value":
        return None
    value = (snak.get("datavalue") or {}).get("value")
    if value is None:
        return None
    if isinstance(value, dict):
        if "id" in value:
            return str(value["id"])
        # Structured values carry more than their headline number. Dropping the
        # unit, the time precision or the coordinate globe made a change from
        # metres to kilometres, from year to day precision, or from Earth to
        # Mars look like no change at all.
        if "amount" in value:
            unit = str(value.get("unit") or "1")
            unit = unit.rsplit("/", 1)[-1] if unit != "1" else "1"
            return f"{value['amount']}|{unit}"
        if "time" in value:
            return (f"{value['time']}|p{value.get('precision', '')}"
                    f"|{str(value.get('calendarmodel') or '').rsplit('/', 1)[-1]}")
        if "latitude" in value and "longitude" in value:
            globe = str(value.get("globe") or "").rsplit("/", 1)[-1]
            return (f"{value['latitude']},{value['longitude']}"
                    f"|p{value.get('precision', '')}|{globe}")
        if "text" in value:
            return f"{value['text']}|{value.get('language', '')}"
        return None
    return str(value)


def value_hash(datatype: str, value: str) -> str:
    """Compact comparison key for an object value."""
    payload = f"{datatype}\x1f{value}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _english_label(entity: Dict[str, Any]) -> str:
    return (((entity.get("labels") or {}).get("en") or {}).get("value") or "").strip()


def _instance_of(entity: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for statement in (entity.get("claims") or {}).get("P31", []) or []:
        value = ((statement.get("mainsnak") or {}).get("datavalue") or {}).get("value")
        if isinstance(value, dict) and "id" in value:
            out.append(str(value["id"]))
    return out


def _wiki_url(sitelinks: Dict[str, Any]) -> str:
    title = ((sitelinks or {}).get("enwiki") or {}).get("title") or ""
    if not title:
        return ""
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def normalise_entity(
    entity: Dict[str, Any],
    *,
    datatypes: Sequence[str],
    require_enwiki: bool,
    apply_entity_filter: bool,
    counts: Dict[str, int],
) -> Optional[Dict[str, Any]]:
    """Normalise one dump entity into ``{meta, statements}`` or ``None``.

    ``counts`` is mutated with drop reasons so the caller can merge funnels
    across worker processes.
    """
    entity_id = entity.get("id") or ""
    kind = entity.get("type")

    if kind == "property":
        # Property entities carry the labels we need for the allow-list.
        label = _english_label(entity)
        if label:
            return {"property_label": (entity_id, label)}
        return None

    if kind != "item" or not entity_id.startswith("Q"):
        counts["entity: not a Q-item"] = counts.get("entity: not a Q-item", 0) + 1
        return None

    label = _english_label(entity)
    sitelinks = entity.get("sitelinks") or {}

    if apply_entity_filter:
        ok, reason = filters.is_allowed_entity(
            label=label,
            sitelinks=sitelinks,
            instance_of=_instance_of(entity),
            require_enwiki=require_enwiki,
        )
        if not ok:
            counts[f"entity: {reason}"] = counts.get(f"entity: {reason}", 0) + 1
            return None

    allowed_types = set(datatypes)
    statements: List[Tuple[str, str, str, str, str]] = []
    for property_id, group in (entity.get("claims") or {}).items():
        if not group:
            continue
        if not filters.is_allowed_relation(property_id=property_id, use_allowlist=False):
            counts["statement: denied predicate"] = counts.get("statement: denied predicate", 0) + len(group)
            continue

        # Every non-deprecated statement is kept, not one representative. With
        # a single pick, a property whose values went from {A, B} to {A, C}
        # could resolve to A in both snapshots and register as unchanged.
        usable = [st for st in group
                  if st.get("rank", "normal") not in filters.EXCLUDED_RANKS]
        if not usable:
            counts["statement: deprecated rank"] = counts.get("statement: deprecated rank", 0) + len(group)
            continue

        for chosen in usable:
            mainsnak = chosen.get("mainsnak") or {}
            datatype = str(mainsnak.get("datatype") or "unknown")
            if datatype not in allowed_types:
                counts["statement: datatype not answerable"] = counts.get("statement: datatype not answerable", 0) + 1
                continue

            value = canonical_value(mainsnak)
            if value is None:
                counts["statement: novalue/somevalue"] = counts.get("statement: novalue/somevalue", 0) + 1
                continue

            statements.append((
                property_id,
                datatype,
                value,
                value_hash(datatype, value),
                str(chosen.get("id") or ""),
            ))

    return {
        "id": entity_id,
        "label": label,
        "wiki_url": _wiki_url(sitelinks),
        "modified": str(entity.get("modified") or ""),
        "has_enwiki": "enwiki" in sitelinks,
        "statements": statements,
    }


def _parse_chunk(payload: Tuple[List[str], Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Parse a chunk of dump lines. Module-level so it survives pickling."""
    lines, options = payload
    counts: Dict[str, int] = {}
    records: List[Dict[str, Any]] = []
    for line in lines:
        try:
            entity = json.loads(line)
        except json.JSONDecodeError:
            counts["entity: unparseable JSON line"] = counts.get("entity: unparseable JSON line", 0) + 1
            continue
        record = normalise_entity(
            entity,
            datatypes=options["datatypes"],
            require_enwiki=options["require_enwiki"],
            apply_entity_filter=options["apply_entity_filter"],
            counts=counts,
        )
        if record is not None:
            records.append(record)
    return records, counts


def _chunked(lines: Iterable[str], size: int) -> Iterator[List[str]]:
    buffer: List[str] = []
    for line in lines:
        buffer.append(line)
        if len(buffer) >= size:
            yield buffer
            buffer = []
    if buffer:
        yield buffer


def iter_records(
    path: Path,
    *,
    options: Dict[str, Any],
    workers: int,
    chunk_size: int,
    limit: Optional[int],
    progress_every: int,
    tag: str,
    stats: filters.FilterStats,
    counter: Optional[Dict[str, int]] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream normalised records from a dump, optionally parsing in parallel.

    Drop reasons are merged into ``stats`` as chunks complete. Parallel mode
    keeps at most ``2 * workers`` chunks in flight so memory stays bounded.
    """
    lines = iter_dump_lines(path, limit=limit)
    chunks = _chunked(lines, chunk_size)
    seen = 0
    started = time.time()

    def _progress() -> None:
        if progress_every > 0 and seen % progress_every == 0:
            rate = seen / max(time.time() - started, 1e-6)
            logger.info("%s: %s entities read (%.0f/s)", tag, f"{seen:,}", rate)

    if workers <= 1:
        for chunk in chunks:
            records, counts = _parse_chunk((chunk, options))
            for reason, n in counts.items():
                stats.drop(reason, n)
            for record in records:
                yield record
            for _ in chunk:
                seen += 1
                _progress()
        if counter is not None:
            counter["lines_read"] = seen
        logger.info("%s: %s entities read in %.1fs", tag, f"{seen:,}", time.time() - started)
        return

    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: deque = deque()
        window = workers * 2
        exhausted = False
        while True:
            while not exhausted and len(pending) < window:
                try:
                    chunk = next(chunks)
                except StopIteration:
                    exhausted = True
                    break
                pending.append((pool.submit(_parse_chunk, (chunk, options)), len(chunk)))
            if not pending:
                break
            future, n_lines = pending.popleft()
            records, counts = future.result()
            for reason, n in counts.items():
                stats.drop(reason, n)
            for record in records:
                yield record
            for _ in range(n_lines):
                seen += 1
                _progress()
    if counter is not None:
        counter["lines_read"] = seen
    logger.info("%s: %s entities read in %.1fs", tag, f"{seen:,}", time.time() - started)


# ---------------------------------------------------------------------------
# On-disk index
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS t0 (
    s TEXT NOT NULL, p TEXT NOT NULL, vhash TEXT NOT NULL, val TEXT NOT NULL,
    -- Keyed by value, not by (s, p): a multivalued property needs a row per
    -- value or a change within the value set is invisible.
    PRIMARY KEY (s, p, vhash)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS delta (
    s TEXT NOT NULL, p TEXT NOT NULL, entity_label TEXT, datatype TEXT,
    old_value TEXT, new_value TEXT, change_type TEXT, ts TEXT, wiki_url TEXT,
    stmt_id TEXT,
    PRIMARY KEY (s, p, new_value)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS labels (
    qid TEXT PRIMARY KEY, label TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS plabels (
    pid TEXT PRIMARY KEY, label TEXT NOT NULL
) WITHOUT ROWID;
"""


def open_index(path: Path) -> sqlite3.Connection:
    """Create the on-disk index. Durability is traded for build speed."""
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-131072")  # ~128 MB page cache
    conn.executescript(SCHEMA)
    return conn


def build_t0_index(
    conn: sqlite3.Connection,
    path: Path,
    *,
    options: Dict[str, Any],
    workers: int,
    chunk_size: int,
    limit: Optional[int],
    progress_every: int,
    stats: filters.FilterStats,
    batch_rows: int = 50_000,
) -> Tuple[int, int, int]:
    """Stream T0 into the ``t0`` table. Returns (lines, entities, statements)."""
    read_counter: Dict[str, int] = {"lines_read": 0}
    entities = 0
    statements = 0
    batch: List[Tuple[str, str, str, str]] = []
    property_labels: List[Tuple[str, str]] = []

    for record in iter_records(
        path, options=options, workers=workers, chunk_size=chunk_size,
        limit=limit, progress_every=progress_every, tag="T0", stats=stats,
        counter=read_counter,
    ):
        if "property_label" in record:
            property_labels.append(record["property_label"])
            continue
        entities += 1
        for property_id, _datatype, value, vhash, _stmt_id in record["statements"]:
            batch.append((record["id"], property_id, vhash, value))
        if len(batch) >= batch_rows:
            conn.executemany("INSERT OR REPLACE INTO t0 VALUES (?, ?, ?, ?)", batch)
            statements += len(batch)
            batch.clear()
            conn.commit()

    if batch:
        conn.executemany("INSERT OR REPLACE INTO t0 VALUES (?, ?, ?, ?)", batch)
        statements += len(batch)
    if property_labels:
        conn.executemany("INSERT OR REPLACE INTO plabels VALUES (?, ?)", property_labels)
    conn.commit()
    return read_counter["lines_read"], entities, statements


def compute_delta(
    conn: sqlite3.Connection,
    path: Path,
    *,
    options: Dict[str, Any],
    workers: int,
    chunk_size: int,
    limit: Optional[int],
    progress_every: int,
    stats: filters.FilterStats,
    fallback_timestamp: str,
    batch_rows: int = 20_000,
) -> Dict[str, int]:
    """Stream T1 against the index, writing candidate deltas to ``delta``."""
    counters = {"lines_read": 0, "entities": 0, "statements": 0,
                "created": 0, "updated": 0, "unchanged": 0}
    read_counter: Dict[str, int] = {"lines_read": 0}
    delta_batch: List[Tuple] = []
    label_batch: List[Tuple[str, str]] = []
    property_labels: List[Tuple[str, str]] = []
    cursor = conn.cursor()

    def flush() -> None:
        if delta_batch:
            conn.executemany(
                "INSERT OR REPLACE INTO delta VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", delta_batch
            )
            delta_batch.clear()
        if label_batch:
            conn.executemany("INSERT OR REPLACE INTO labels VALUES (?, ?)", label_batch)
            label_batch.clear()
        conn.commit()

    for record in iter_records(
        path, options=options, workers=workers, chunk_size=chunk_size,
        limit=limit, progress_every=progress_every, tag="T1", stats=stats,
        counter=read_counter,
    ):
        if "property_label" in record:
            property_labels.append(record["property_label"])
            continue

        counters["entities"] += 1
        if record["label"]:
            label_batch.append((record["id"], record["label"]))

        timestamp = record["modified"] or fallback_timestamp
        # Cache the T0 value set per (s, p) so a multivalued property costs one
        # query rather than one per value.
        t0_sets: Dict[str, Tuple[set, list]] = {}
        for property_id, datatype, value, vhash, stmt_id in record["statements"]:
            counters["statements"] += 1
            if property_id not in t0_sets:
                cursor.execute("SELECT vhash, val FROM t0 WHERE s = ? AND p = ?",
                               (record["id"], property_id))
                rows = cursor.fetchall()
                t0_sets[property_id] = ({r[0] for r in rows}, [r[1] for r in rows])
            old_hashes, old_values = t0_sets[property_id]

            if not old_hashes:
                # Delta_plus: the (subject, relation) pair is new in T1.
                change_type, old_value = "created", NEW_CREATED
                counters["created"] += 1
            elif vhash in old_hashes:
                counters["unchanged"] += 1
                stats.drop("statement: unchanged between T0 and T1", 1)
                continue
            else:
                # Delta_circle: the pair existed, but this value is new to it.
                change_type = "updated"
                old_value = old_values[0] if len(old_values) == 1 else "|".join(sorted(old_values))
                counters["updated"] += 1

            delta_batch.append((
                record["id"], property_id, record["label"], datatype, old_value, value,
                change_type, timestamp, record["wiki_url"], stmt_id,
            ))

        if len(delta_batch) >= batch_rows or len(label_batch) >= batch_rows:
            flush()

    if property_labels:
        conn.executemany("INSERT OR REPLACE INTO plabels VALUES (?, ?)", property_labels)
    flush()
    counters["lines_read"] = read_counter["lines_read"]
    return counters


# ---------------------------------------------------------------------------
# Label resolution and CSV emission
# ---------------------------------------------------------------------------

def load_property_labels(conn: sqlite3.Connection, extra: Optional[Path]) -> Dict[str, str]:
    """Property labels from the dumps, overridden by ``--property-labels``."""
    labels = {pid: label for pid, label in conn.execute("SELECT pid, label FROM plabels")}
    if extra is not None:
        payload = json.loads(extra.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{extra} must contain a JSON object mapping property IDs to labels")
        labels.update({str(k): str(v) for k, v in payload.items()})
    return labels


def fetch_missing_labels(ids: Sequence[str], *, component: str) -> Dict[str, str]:
    """Resolve labels from the Wikidata Action API in batches of 50."""
    if not ids:
        return {}
    from livesearchbench.http import PoliteSession, RequestFailed

    resolved: Dict[str, str] = {}
    with PoliteSession(component=component, min_interval=0.2) as session:
        for start in range(0, len(ids), 50):
            chunk = list(ids[start:start + 50])
            try:
                data = session.wikidata_api({
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "labels",
                    "languages": "en",
                })
            except RequestFailed as exc:
                # Labels are cosmetic; a hard failure here must not discard a
                # completed delta, but it must be visible.
                logger.warning("Label lookup failed for %d ids: %s", len(chunk), exc)
                continue
            for entity_id, entity in (data.get("entities") or {}).items():
                label = (((entity.get("labels") or {}).get("en") or {}).get("value") or "").strip()
                if label:
                    resolved[entity_id] = label
    return resolved


def emit_csv(
    conn: sqlite3.Connection,
    output: Path,
    *,
    property_labels: Dict[str, str],
    use_allowlist: bool,
    offline: bool,
    stats: filters.FilterStats,
) -> Dict[str, int]:
    """Join labels onto the candidate deltas and write the CSV."""
    counters = {"candidates": 0, "allowed": 0, "written": 0}
    seen: set = set()
    rows: List[List[str]] = []

    query = "SELECT s, p, entity_label, datatype, old_value, new_value, change_type, ts, wiki_url, stmt_id FROM delta"
    for s, p, entity_label, datatype, old_value, new_value, change_type, ts, wiki_url, stmt_id in conn.execute(query):
        counters["candidates"] += 1
        property_label = property_labels.get(p, p)
        if not filters.is_allowed_relation(
            property_id=p, property_label=property_label, use_allowlist=use_allowlist
        ):
            stats.drop("relation: not in allow-list", 1)
            continue
        key = filters.dedup_key(s, p, stmt_id)
        if key in seen:
            stats.drop("statement: duplicate (s, r)", 1)
            continue
        seen.add(key)
        counters["allowed"] += 1
        rows.append([s, entity_label or s, p, property_label, datatype,
                     old_value, new_value, "", change_type, ts, wiki_url])

    # Resolve object labels: dump-derived table first, API for the remainder.
    needed = {row[6] for row in rows if row[6].startswith("Q") and row[6][1:].isdigit()}
    object_labels: Dict[str, str] = {}
    if needed:
        placeholders = list(needed)
        for start in range(0, len(placeholders), 900):
            chunk = placeholders[start:start + 900]
            marks = ",".join("?" * len(chunk))
            for qid, label in conn.execute(f"SELECT qid, label FROM labels WHERE qid IN ({marks})", chunk):
                object_labels[qid] = label
        missing = sorted(needed - set(object_labels))
        if missing and not offline:
            logger.info("Resolving %d object labels missing from the dump via the Wikidata API", len(missing))
            object_labels.update(fetch_missing_labels(missing, component="LiveSearchBench-dump-delta"))
        elif missing:
            logger.warning("%d object labels unresolved (--offline); falling back to the QID", len(missing))

    for row in rows:
        row[7] = object_labels.get(row[6], row[6])

    rows.sort(key=lambda r: (r[9], r[0]), reverse=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    counters["written"] = len(rows)
    return counters


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def parse_dump_date(name: str) -> Optional[str]:
    """Extract a dump date such as ``wikidata-20210104-all.json.gz``."""
    match = _DUMP_DATE_RE.search(name)
    if not match:
        return None
    return "-".join(match.groups())


def sha256_file(path: Path, *, block: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_dump(path: Path, *, skip_hash: bool) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "filename": path.name,
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": None if skip_hash else sha256_file(path),
        "dump_date": parse_dump_date(path.name),
    }


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

def _item(qid, label, claims, *, enwiki=None, dewiki=None, modified="2021-01-04T00:00:00Z",
          description="synthetic fixture entity"):
    sitelinks = {}
    if enwiki:
        sitelinks["enwiki"] = {"site": "enwiki", "title": enwiki, "badges": []}
    if dewiki:
        sitelinks["dewiki"] = {"site": "dewiki", "title": dewiki, "badges": []}
    return {
        "type": "item",
        "id": qid,
        "labels": {"en": {"language": "en", "value": label}},
        "descriptions": {"en": {"language": "en", "value": description}},
        "aliases": {},
        "claims": claims,
        "sitelinks": sitelinks,
        "lastrevid": 1,
        "modified": modified,
    }


def _claim(qid, property_id, value, *, datatype="wikibase-item", rank="normal", suffix="a"):
    if datatype == "wikibase-item":
        datavalue = {"value": {"entity-type": "item", "numeric-id": int(value[1:]), "id": value},
                     "type": "wikibase-entityid"}
    elif datatype == "commonsMedia":
        datavalue = {"value": value, "type": "string"}
    else:
        datavalue = {"value": value, "type": "string"}
    return {
        "mainsnak": {"snaktype": "value", "property": property_id,
                     "datavalue": datavalue, "datatype": datatype},
        "type": "statement",
        "id": f"{qid}${property_id}-{suffix}",
        "rank": rank,
    }


def _property(pid, label, datatype):
    return {
        "type": "property",
        "id": pid,
        "datatype": datatype,
        "labels": {"en": {"language": "en", "value": label}},
        "descriptions": {"en": {"language": "en", "value": f"fixture property {pid}"}},
        "aliases": {},
        "claims": {},
    }


def _objects(modified):
    """Shared object entities, so labels resolve without network access."""
    known = [
        ("Q5", "human", "Human"), ("Q84", "London", "London"), ("Q90", "Paris", "Paris"),
        ("Q145", "United Kingdom", "United Kingdom"), ("Q30", "United States", "United States"),
        ("Q312", "Apple Inc.", "Apple Inc."), ("Q60", "New York City", "New York City"),
        ("Q1234", "Example Football Club", "Example F.C."), ("Q183", "Germany", "Germany"),
    ]
    out = [_item(qid, label, {}, enwiki=title, modified=modified) for qid, label, title in known]
    out.append(_item("Q4167410", "Wikimedia disambiguation page", {}, modified=modified))
    return out


def _quantity_claim(qid, property_id, amount, *, unit="1", rank="normal", suffix="q"):
    """A quantity statement, including its unit."""
    unit_uri = "1" if unit == "1" else f"http://www.wikidata.org/entity/{unit}"
    return {
        "id": f"{qid}${property_id}-{suffix}",
        "rank": rank,
        "type": "statement",
        "mainsnak": {
            "snaktype": "value", "property": property_id, "datatype": "quantity",
            "datavalue": {"type": "quantity",
                          "value": {"amount": amount, "unit": unit_uri}},
        },
    }


def _time_claim(qid, property_id, time_value, *, precision=11, rank="normal", suffix="t"):
    """A time statement, including its precision and calendar model."""
    return {
        "id": f"{qid}${property_id}-{suffix}",
        "rank": rank,
        "type": "statement",
        "mainsnak": {
            "snaktype": "value", "property": property_id, "datatype": "time",
            "datavalue": {"type": "time",
                          "value": {"time": time_value, "precision": precision,
                                    "calendarmodel": "http://www.wikidata.org/entity/Q1985727"}},
        },
    }


def build_fixture_entities(which: str) -> List[Dict[str, Any]]:
    """Entities for the T0 or T1 synthetic snapshot. See the module docstring."""
    modified = "2021-01-04T00:00:00Z" if which == "T0" else "2025-01-06T00:00:00Z"
    properties = [
        _property("P17", "country", "wikibase-item"),
        _property("P19", "place of birth", "wikibase-item"),
        _property("P176", "manufacturer", "wikibase-item"),
        _property("P54", "member of sports team", "wikibase-item"),
        _property("P18", "image", "commonsMedia"),
        _property("P31", "instance of", "wikibase-item"),
        _property("P106", "occupation", "wikibase-item"),
        _property("P2046", "area", "quantity"),
        _property("P569", "date of birth", "time"),
    ]

    if which == "T0":
        q42 = _item("Q42", "Ada Fixture", {
            "P31": [_claim("Q42", "P31", "Q5")],
            "P19": [_claim("Q42", "P19", "Q84")],
            "P17": [_claim("Q42", "P17", "Q145")],
            "P18": [_claim("Q42", "P18", "Old_portrait.jpg", datatype="commonsMedia")],
            # Multivalued: the set {Q901, Q902} becomes {Q901, Q903} in T1.
            "P106": [_claim("Q42", "P106", "Q901"), _claim("Q42", "P106", "Q902", suffix="b")],
            # Same amount, different unit: metres in T0, kilometres in T1.
            "P2046": [_quantity_claim("Q42", "P2046", "+100", unit="Q11573")],
            # Same instant, coarser precision in T1.
            "P569": [_time_claim("Q42", "P569", "+1815-12-10T00:00:00Z", precision=11)],
        }, enwiki="Ada Fixture", modified=modified)
        q7259 = _item("Q7259", "Fixture Devices Ltd", {
            "P31": [_claim("Q7259", "P31", "Q5")],
            "P17": [_claim("Q7259", "P17", "Q30")],
        }, enwiki="Fixture Devices Ltd", modified=modified)
        q999 = _item("Q999", "Kein Wiki Eintrag", {
            "P17": [_claim("Q999", "P17", "Q145")],
        }, dewiki="Kein Wiki Eintrag", modified=modified)
        q31000 = _item("Q31000", "Fixture (disambiguation)", {
            "P31": [_claim("Q31000", "P31", "Q4167410")],
            "P17": [_claim("Q31000", "P17", "Q145")],
        }, enwiki="Fixture (disambiguation)", modified=modified)
        entities = [q42, q7259, q999, q31000]
    else:
        q42 = _item("Q42", "Ada Fixture", {
            "P31": [_claim("Q42", "P31", "Q5")],
            # Delta_circle: the object of an existing (s, r) changed.
            "P19": [_claim("Q42", "P19", "Q90", suffix="b")],
            # Unchanged: identical object, must not appear in the delta.
            "P17": [_claim("Q42", "P17", "Q145")],
            # Denied predicate: changed, but P18 is on the deny-list.
            "P18": [_claim("Q42", "P18", "New_portrait.jpg", datatype="commonsMedia", suffix="b")],
            # One value of the set changed; the other is untouched.
            "P106": [_claim("Q42", "P106", "Q901"), _claim("Q42", "P106", "Q903", suffix="c")],
            # Unit change only. With the amount alone this looked unchanged.
            "P2046": [_quantity_claim("Q42", "P2046", "+100", unit="Q828224")],
            # Precision change only. With the timestamp alone this looked unchanged.
            "P569": [_time_claim("Q42", "P569", "+1815-12-10T00:00:00Z", precision=9)],
        }, enwiki="Ada Fixture", modified=modified)
        q7259 = _item("Q7259", "Fixture Devices Ltd", {
            "P31": [_claim("Q7259", "P31", "Q5")],
            "P17": [_claim("Q7259", "P17", "Q30")],
            # Delta_plus: a (s, r) pair absent from T0.
            "P176": [_claim("Q7259", "P176", "Q312")],
            # Relation outside the 198-label allow-list.
            "P54": [_claim("Q7259", "P54", "Q1234")],
            # Only statement for this (s, r) and it is deprecated.
            "P19": [_claim("Q7259", "P19", "Q60", rank="deprecated")],
        }, enwiki="Fixture Devices Ltd", modified=modified)
        q999 = _item("Q999", "Kein Wiki Eintrag", {
            # New triple, but the entity has no English Wikipedia sitelink.
            "P17": [_claim("Q999", "P17", "Q183", suffix="b")],
        }, dewiki="Kein Wiki Eintrag", modified=modified)
        q31000 = _item("Q31000", "Fixture (disambiguation)", {
            "P31": [_claim("Q31000", "P31", "Q4167410")],
            # New triple on a Wikimedia disambiguation page.
            "P17": [_claim("Q31000", "P17", "Q183", suffix="b")],
        }, enwiki="Fixture (disambiguation)", modified=modified)
        entities = [q42, q7259, q999, q31000]

    return properties + entities + _objects(modified)


def write_fixture(directory: Path) -> List[Path]:
    """Write ``dump_T0.json.gz`` and ``dump_T1.json.gz`` into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for which in ("T0", "T1"):
        path = directory / f"dump_{which}.json.gz"
        entities = build_fixture_entities(which)
        # mtime=0 keeps the gzip byte stream reproducible across runs.
        with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as raw:
            raw.write(b"[\n")
            for index, entity in enumerate(entities):
                suffix = ",\n" if index < len(entities) - 1 else "\n"
                raw.write((json.dumps(entity, ensure_ascii=False) + suffix).encode("utf-8"))
            raw.write(b"]\n")
        logger.info("Wrote fixture %s (%d entities)", path, len(entities))
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_dump_delta.py",
        description=(
            "Step 1: compute the paper's Delta = Delta_plus (insertions) union "
            "Delta_circle (object updates) between two Wikidata JSON dumps, and "
            "write the CSV that generate_level{1,2,3}.py consume."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/extract_dump_delta.py --make-fixture data/sample\n"
            "  python scripts/extract_dump_delta.py --t0 data/sample/dump_T0.json.gz \\\n"
            "      --t1 data/sample/dump_T1.json.gz \\\n"
            "      --output outputs/extracted_triples/delta_sample.csv --offline\n"
        ),
    )
    parser.add_argument("--t0", type=Path,
                        help="Wikidata JSON dump for the earlier snapshot (.json/.json.gz/.json.bz2)")
    parser.add_argument("--t1", type=Path,
                        help="Wikidata JSON dump for the later snapshot (.json/.json.gz/.json.bz2)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: outputs/extracted_triples/dump_delta_<T0>_<T1>.csv)")
    parser.add_argument("--stats-json", type=Path, default=None,
                        help="Filter-funnel JSON (default: <output stem>_filter_stats.json)")
    parser.add_argument("--stats-md", type=Path, default=None,
                        help="Filter-funnel markdown table (default: <output stem>_filter_stats.md)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Provenance manifest JSON (default: <output stem>_manifest.json)")
    parser.add_argument("--index-db", type=Path, default=None,
                        help="Path for the on-disk sqlite index (default: <output stem>.index.sqlite3)")
    parser.add_argument("--keep-index", action="store_true",
                        help="Keep the sqlite index after the run instead of deleting it")
    parser.add_argument("--limit", type=int, default=None,
                        help="Read at most N entity lines from each dump (quick partial run)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Processes used to parse dump lines (default: 1; 4-8 helps on real dumps)")
    parser.add_argument("--chunk-size", type=int, default=2000,
                        help="Dump lines handed to a worker at a time (default: 2000)")
    parser.add_argument("--progress-every", type=int, default=100_000,
                        help="Log progress every N entities read; 0 disables (default: 100000)")
    parser.add_argument("--skip-hash", action="store_true",
                        help="Do not sha256 the dumps for the manifest (much faster on ~100GB files)")
    parser.add_argument("--datatypes", type=str, default=",".join(DEFAULT_DATATYPES),
                        help="Comma-separated property datatypes kept (default: %(default)s)")
    parser.add_argument("--no-allowlist", action="store_true",
                        help="Skip the 198-label relation allow-list; the property-ID deny-list still applies")
    parser.add_argument("--no-require-enwiki", action="store_true",
                        help="Keep T1 subjects without an English Wikipedia sitelink")
    parser.add_argument("--index-all-t0", dest="index_all_t0", action="store_true", default=True,
                        help="Index every T0 entity (default): avoids false insertions for entities "
                             "that only gained a sitelink between the snapshots")
    parser.add_argument("--filter-t0-entities", dest="index_all_t0", action="store_false",
                        help="Apply the entity filter to T0 as well, for a much smaller index")
    parser.add_argument("--property-labels", type=Path, default=None,
                        help="JSON object {property_id: label} overriding labels found in the dumps")
    parser.add_argument("--offline", action="store_true",
                        help="Never call the Wikidata API; unresolved labels fall back to the QID")
    parser.add_argument("--make-fixture", type=Path, default=None, metavar="DIR",
                        help="Write the synthetic dump_T0/dump_T1 fixture into DIR and exit")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")
    return parser


def default_output(t0: Path, t1: Path) -> Path:
    left = parse_dump_date(t0.name) or t0.stem.split(".")[0]
    right = parse_dump_date(t1.name) or t1.stem.split(".")[0]
    return PROJECT_ROOT / "outputs" / "extracted_triples" / f"dump_delta_{left}_to_{right}.csv"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.make_fixture is not None:
        write_fixture(args.make_fixture)
        return 0

    # -- validate everything before any long work starts ---------------------
    if args.t0 is None or args.t1 is None:
        parser.error("--t0 and --t1 are required (or use --make-fixture DIR)")
    for label, path in (("--t0", args.t0), ("--t1", args.t1)):
        if not path.is_file():
            parser.error(f"{label}: no such file: {path}")
        try:
            open_dump(path).close()  # fails now, not after an hour of work
        except (ValueError, OSError) as exc:
            parser.error(f"{label}: {exc}")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    datatypes = tuple(x.strip() for x in args.datatypes.split(",") if x.strip())
    if not datatypes:
        parser.error("--datatypes must name at least one datatype")
    if args.property_labels is not None and not args.property_labels.is_file():
        parser.error(f"--property-labels: no such file: {args.property_labels}")

    output = args.output or default_output(args.t0, args.t1)
    stem = output.with_suffix("")
    stats_json = args.stats_json or Path(f"{stem}_filter_stats.json")
    stats_md = args.stats_md or Path(f"{stem}_filter_stats.md")
    manifest_path = args.manifest or Path(f"{stem}_manifest.json")
    index_db = args.index_db or Path(f"{stem}.index.sqlite3")
    for path in (output, stats_json, stats_md, manifest_path, index_db):
        path.parent.mkdir(parents=True, exist_ok=True)
    if index_db.exists():
        index_db.unlink()

    started = time.time()
    t0_stats = filters.FilterStats()
    t1_stats = filters.FilterStats()

    t0_options = {
        "datatypes": list(datatypes),
        "require_enwiki": not args.no_require_enwiki,
        "apply_entity_filter": not args.index_all_t0,
    }
    t1_options = dict(t0_options, apply_entity_filter=True)

    logger.info("Indexing T0 %s -> %s", args.t0.name, index_db)
    conn = open_index(index_db)
    try:
        t0_lines, t0_entities, t0_statements = build_t0_index(
            conn, args.t0, options=t0_options, workers=args.workers,
            chunk_size=args.chunk_size, limit=args.limit,
            progress_every=args.progress_every, stats=t0_stats,
        )
        t0_stats.stage("T0 entity lines read", t0_lines)
        t0_stats.stage("T0 items indexed", t0_entities)
        t0_stats.stage("T0 statements indexed", t0_statements)
        logger.info("T0 index: %s entities, %s statements", f"{t0_entities:,}", f"{t0_statements:,}")

        fallback_timestamp = ""
        t1_date = parse_dump_date(args.t1.name)
        if t1_date:
            fallback_timestamp = f"{t1_date}T00:00:00Z"

        logger.info("Streaming T1 %s against the index", args.t1.name)
        delta_counts = compute_delta(
            conn, args.t1, options=t1_options, workers=args.workers,
            chunk_size=args.chunk_size, limit=args.limit,
            progress_every=args.progress_every, stats=t1_stats,
            fallback_timestamp=fallback_timestamp,
        )
        t1_stats.stage("T1 entity lines read", delta_counts["lines_read"])
        t1_stats.stage("T1 items passing the entity filter", delta_counts["entities"])
        t1_stats.stage("T1 statements after rank/deny/datatype filters", delta_counts["statements"])
        t1_stats.stage(
            "candidate delta (Delta_plus + Delta_circle)",
            delta_counts["created"] + delta_counts["updated"],
        )

        property_labels = load_property_labels(conn, args.property_labels)
        logger.info("Resolved %d property labels from the dumps/overrides", len(property_labels))
        emit_counts = emit_csv(
            conn, output,
            property_labels=property_labels,
            use_allowlist=not args.no_allowlist,
            offline=args.offline,
            stats=t1_stats,
        )
        t1_stats.stage("after relation allow-list and (s, r) dedup", emit_counts["allowed"])
        t1_stats.stage("rows written", emit_counts["written"])
    finally:
        conn.close()
        if not args.keep_index and index_db.exists():
            index_db.unlink()

    elapsed = time.time() - started
    funnel = {
        "t0": t0_stats.to_dict(),
        "t1": t1_stats.to_dict(),
        "delta": {
            "delta_plus_created": delta_counts["created"],
            "delta_circle_updated": delta_counts["updated"],
            "unchanged": delta_counts["unchanged"],
            "rows_written": emit_counts["written"],
        },
    }
    stats_json.write_text(json.dumps(funnel, indent=2) + "\n", encoding="utf-8")
    stats_md.write_text(
        "# Dump-differential filter funnel\n\n"
        f"T0: `{args.t0.name}`  \nT1: `{args.t1.name}`\n\n"
        "## T0 indexing\n\n" + t0_stats.render() + "\n\n"
        "## T1 differencing\n\n" + t1_stats.render() + "\n\n"
        "## Delta\n\n"
        "| Component | Count |\n|---|---:|\n"
        f"| Delta_plus (created) | {delta_counts['created']:,} |\n"
        f"| Delta_circle (updated) | {delta_counts['updated']:,} |\n"
        f"| unchanged (dropped) | {delta_counts['unchanged']:,} |\n"
        f"| rows written | {emit_counts['written']:,} |\n",
        encoding="utf-8",
    )

    manifest = {
        "tool": "scripts/extract_dump_delta.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 2),
        "python": sys.version.split()[0],
        "dumps": {
            "T0": describe_dump(args.t0, skip_hash=args.skip_hash),
            "T1": describe_dump(args.t1, skip_hash=args.skip_hash),
        },
        "parameters": {
            "limit": args.limit,
            "workers": args.workers,
            "chunk_size": args.chunk_size,
            "datatypes": list(datatypes),
            "use_relation_allowlist": not args.no_allowlist,
            "require_enwiki": not args.no_require_enwiki,
            "index_all_t0": args.index_all_t0,
            "offline": args.offline,
            "skip_hash": args.skip_hash,
        },
        "counts": funnel["delta"] | {
            "t0_entities": t0_entities,
            "t0_statements_indexed": t0_statements,
            "t1_entities": delta_counts["entities"],
            "t1_statements": delta_counts["statements"],
        },
        "outputs": {
            "csv": str(output.resolve()),
            "filter_stats_json": str(stats_json.resolve()),
            "filter_stats_md": str(stats_md.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    logger.info("Delta_plus=%d  Delta_circle=%d  unchanged=%d  rows=%d",
                delta_counts["created"], delta_counts["updated"],
                delta_counts["unchanged"], emit_counts["written"])
    logger.info("CSV      %s", output)
    logger.info("Funnel   %s / %s", stats_json, stats_md)
    logger.info("Manifest %s", manifest_path)
    logger.info("Done in %.1fs", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
