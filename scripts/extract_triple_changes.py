"""Step 0: extract knowledge triple changes from the Wikidata live feed.

Scans ``list=recentchanges`` backwards from now, resolves every candidate edit
against the old and new revision JSON, applies the paper's Step-2 filters
(relation allow-list, entity quality, statement rank) and streams the surviving
(subject, relation, old, new) rows to a CSV that ``generate_level*.py`` consume.

What changed relative to the first release:

* All HTTP now goes through :class:`livesearchbench.http.PoliteSession`:
  bounded retries, exponential backoff, a compliant User-Agent with contact
  info, ``maxlag``, and a timeout on *every* request. The previous version
  retried an ``internal_api_error_DBQueryTimeoutError`` forever in a
  ``while True: sleep(5)`` loop and never wrote a single row.
* ``rclimit`` defaults to 200 instead of 500. 500 is what provoked the
  database timeouts in the first place; ``--rc-limit`` exposes it.
* The CSV is created, header-written and flushed *incrementally*, and its
  parent directory is created before the scan starts, so a 90-minute run that
  is interrupted still leaves usable output.
* ``--hours`` is validated against the ~30-day recentchanges retention window
  instead of silently returning nothing.

Examples::

    python scripts/extract_triple_changes.py --hours 2
    python scripts/extract_triple_changes.py --hours 0.05 --max-triples 5 --no-allowlist
    python scripts/extract_triple_changes.py --hours 6 --change-type all --resume \
        --output outputs/extracted_triples/triple_changes.csv
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import re
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from livesearchbench import config, dataio, filters  # noqa: E402
from livesearchbench.http import PoliteSession, RequestFailed  # noqa: E402

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXTRACTED_TRIPLES_DIR = OUTPUTS_DIR / "extracted_triples"

logger = logging.getLogger("triple_extractor")

#: CSV header. Frozen: scripts/generate_level{1,2,3}.py index these names.
CSV_HEADER = [
    "entity_id", "entity_label", "property_id", "property_label", "property_type",
    "old_value", "new_value", "new_value_label", "change_type", "change_timestamp", "wiki_url",
]
_COL = {name: i for i, name in enumerate(CSV_HEADER)}

#: Wikidata's recentchanges table is pruned after roughly 30 days.
RECENTCHANGES_RETENTION_HOURS = 30 * 24

#: wbgetentities / revids accept 50 ids per call for anonymous clients.
BATCH_SIZE = 50

#: Object datatypes that yield answerable questions.
ALLOWED_TYPES = {"time", "quantity", "wikibase-item", "globe-coordinate"}

CHANGE_TYPES = ("created", "updated", "deleted")
DEFAULT_CHANGE_TYPES = "created,updated"

_PROPERTY_IN_COMMENT = re.compile(r"\[\[Property:(P\d+)\]\]")

#: Sentinels written into the old_value/new_value columns.
_ERROR = "ERROR"
_NO_VALUE = "NO_VALUE"
_SOME_VALUE = "SOME_VALUE"
_NEW_CREATED = "NEW_CREATED"


def get_timestamp_str(dt: datetime) -> str:
    """Format a naive UTC datetime as a Wikidata timestamp string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_change_types(raw: str) -> Set[str]:
    """Parse the comma-separated ``--change-type`` value into a set."""
    wanted = {part.strip().lower() for part in raw.split(",") if part.strip()}
    if not wanted:
        raise argparse.ArgumentTypeError("--change-type may not be empty")
    if "all" in wanted:
        return set(CHANGE_TYPES)
    unknown = wanted - set(CHANGE_TYPES)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown change type(s): {', '.join(sorted(unknown))}. "
            f"Choose from {', '.join(CHANGE_TYPES)}, all."
        )
    return wanted


class CsvSink:
    """Incremental CSV writer that flushes after every batch.

    The previous release buffered everything in memory and wrote once at the
    very end, so an interrupted (or crashed) 90-minute scan produced no file
    at all -- and never created the parent directory of ``--output``.
    """

    def __init__(self, path: str, *, resume: bool = False) -> None:
        self.path = dataio.ensure_parent(Path(path))
        self.seen_entities: Set[str] = set()
        exists = self.path.is_file() and self.path.stat().st_size > 0

        if resume and exists:
            self.seen_entities = self._read_existing_entities()
            self._handle = self.path.open("a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._handle)
            logger.info("Resuming: %d entities already present in %s",
                        len(self.seen_entities), self.path)
        else:
            if resume:
                logger.info("--resume requested but %s does not exist yet; starting fresh", self.path)
            self._handle = self.path.open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._handle)
            self._writer.writerow(CSV_HEADER)
            self._handle.flush()
        self.rows_written = 0

    def _read_existing_entities(self) -> Set[str]:
        with self.path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            try:
                header = next(reader)
            except StopIteration:
                return set()
            if header != CSV_HEADER:
                raise SystemExit(
                    f"Cannot --resume {self.path}: its header does not match the expected columns.\n"
                    f"  expected: {','.join(CSV_HEADER)}\n"
                    f"  found:    {','.join(header)}"
                )
            return {row[0] for row in reader if row}

    def write_rows(self, rows: Sequence[Sequence[str]]) -> int:
        """Append rows, skipping entities already written. Returns the count."""
        added = 0
        for row in rows:
            entity_id = row[0]
            if entity_id in self.seen_entities:
                continue
            self.seen_entities.add(entity_id)
            self._writer.writerow(row)
            added += 1
        if added:
            self._handle.flush()
            self.rows_written += added
        return added

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()


class TripleChangeExtractor:
    """Extract knowledge triple changes from the Wikidata live feed."""

    def __init__(
        self,
        *,
        hours: Optional[float] = None,
        output_file: str = "triple_changes.csv",
        max_triples: Optional[int] = None,
        rc_limit: int = 200,
        change_types: Optional[Set[str]] = None,
        use_allowlist: bool = True,
        max_workers: int = 10,
        resume: bool = False,
        api_endpoint: Optional[str] = None,
        max_attempts: int = 5,
        min_interval: float = 0.0,
        require_enwiki: bool = True,
    ) -> None:
        self.end_time = datetime.now(timezone.utc).replace(tzinfo=None)
        self.start_time = self.end_time - timedelta(hours=hours) if hours is not None else None
        self.hours = hours
        self.output_file = output_file
        self.max_triples = max_triples
        self.rc_limit = max(1, int(rc_limit))
        self.change_types = set(change_types or parse_change_types(DEFAULT_CHANGE_TYPES))
        self.use_allowlist = use_allowlist
        self.max_workers = max(1, int(max_workers))
        self.resume = resume
        self.api_endpoint = api_endpoint or config.get("WIKIDATA_API", default=config.DEFAULT_WIKIDATA_API)
        self.max_attempts = max(1, int(max_attempts))
        self.min_interval = float(min_interval)
        self.require_enwiki = require_enwiki

        self.total_changes = 0
        self.kept_triples = 0
        self.stats = filters.FilterStats()
        self._counts: Dict[str, int] = {key: 0 for key in (
            "scanned", "item_edits", "predicate_in_comment",
            "entity_passed", "relation_passed", "diff_resolved",
            "change_type_in_scope", "written",
        )}
        self._local = threading.local()
        self._sessions: List[PoliteSession] = []
        self._sessions_lock = threading.Lock()

    # -- HTTP --------------------------------------------------------------

    def session(self) -> PoliteSession:
        """Return this thread's :class:`PoliteSession`.

        ``requests.Session`` is not documented as thread-safe and
        ``PoliteSession`` keeps unsynchronised throttling state, so each
        worker thread gets its own instead of sharing one.
        """
        existing = getattr(self._local, "session", None)
        if existing is not None:
            return existing
        created = PoliteSession(
            component="LiveSearchBench-extractor",
            max_attempts=self.max_attempts,
            min_interval=self.min_interval,
        )
        self._local.session = created
        with self._sessions_lock:
            self._sessions.append(created)
        return created

    def close_sessions(self) -> None:
        with self._sessions_lock:
            sessions, self._sessions = self._sessions, []
        for session in sessions:
            session.close()

    def _api(self, params: Dict[str, object]) -> Dict:
        return self.session().wikidata_api(params, endpoint=self.api_endpoint)

    # -- recentchanges -----------------------------------------------------

    def fetch_recent_changes_generator(self) -> Iterable[List[Dict]]:
        """Yield batches of recentchanges records, newest first."""
        params: Dict[str, object] = {
            "action": "query",
            "list": "recentchanges",
            "rcnamespace": 0,
            "rcprop": "ids|title|timestamp|comment",
            "rctype": "edit|new",
            "rclimit": self.rc_limit,
            "rcstart": get_timestamp_str(self.end_time),
            "formatversion": "2",
        }
        if self.start_time is not None:
            params["rcend"] = get_timestamp_str(self.start_time)

        while True:
            data = self._api(params)
            changes = data.get("query", {}).get("recentchanges", [])
            if not changes:
                return
            yield changes
            continuation = data.get("continue")
            if not continuation:
                return
            params.update(continuation)

    # -- label helpers -----------------------------------------------------

    def _fetch_labels(self, ids: Sequence[str]) -> Dict[str, str]:
        """Fetch English labels for a list of Q/P ids, falling back to the id."""
        if not ids:
            return {}
        labels: Dict[str, str] = {}
        for start in range(0, len(ids), BATCH_SIZE):
            chunk = list(ids[start:start + BATCH_SIZE])
            try:
                data = self._api({
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "props": "labels",
                    "languages": "en",
                })
            except RequestFailed as exc:
                # A missing label degrades a row, it does not invalidate it, so
                # this is the one place a failure falls back instead of raising.
                logger.warning("Label lookup failed for %d ids (%s); using bare ids", len(chunk), exc)
                labels.update({identifier: identifier for identifier in chunk})
                continue
            entities = data.get("entities", {})
            for identifier, payload in entities.items():
                label = payload.get("labels", {}).get("en", {}).get("value")
                labels[identifier] = label or identifier
            for identifier in chunk:
                labels.setdefault(identifier, identifier)
        return labels

    # -- entity + revision fetching ---------------------------------------

    def _fetch_entities(self, qids: Sequence[str]) -> Dict[str, dict]:
        chunks = [list(qids[i:i + BATCH_SIZE]) for i in range(0, len(qids), BATCH_SIZE)]

        def fetch(chunk: List[str]) -> Dict[str, dict]:
            data = self._api({
                "action": "wbgetentities",
                "ids": "|".join(chunk),
                "props": "sitelinks|claims|labels",
                "languages": "en",
                "sitefilter": "enwiki",
            })
            return data.get("entities", {})

        return self._map_chunks(fetch, chunks, "entity")

    def _batch_fetch_revisions(self, revids: Sequence[str]) -> Dict[str, Optional[dict]]:
        """Fetch and JSON-decode revision content, keyed by revid."""
        if not revids:
            return {}
        chunks = [list(revids[i:i + BATCH_SIZE]) for i in range(0, len(revids), BATCH_SIZE)]

        def fetch(chunk: List[str]) -> Dict[str, Optional[dict]]:
            data = self._api({
                "action": "query",
                "prop": "revisions",
                "revids": "|".join(chunk),
                "rvprop": "ids|content",
                "rvslots": "main",
                "formatversion": "2",
            })
            results: Dict[str, Optional[dict]] = {}
            for page in data.get("query", {}).get("pages", []):
                for revision in page.get("revisions", []):
                    revid = str(revision.get("revid", ""))
                    content = revision.get("slots", {}).get("main", {}).get("content")
                    if not content:
                        continue
                    if isinstance(content, str):
                        try:
                            results[revid] = json.loads(content)
                        except json.JSONDecodeError:
                            logger.warning("Revision %s has non-JSON content; skipping", revid)
                            results[revid] = None
                    else:
                        results[revid] = content
            return results

        return self._map_chunks(fetch, chunks, "revision")

    def _map_chunks(self, fetch, chunks: List[List[str]], what: str) -> Dict:
        """Run ``fetch`` over chunks with the configured worker count."""
        merged: Dict = {}
        if not chunks:
            return merged
        if self.max_workers == 1 or len(chunks) == 1:
            for chunk in chunks:
                merged.update(self._safe_fetch(fetch, chunk, what))
            return merged
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(self._safe_fetch, fetch, chunk, what) for chunk in chunks]
            for future in concurrent.futures.as_completed(futures):
                merged.update(future.result())
        return merged

    def _safe_fetch(self, fetch, chunk: List[str], what: str) -> Dict:
        try:
            return fetch(chunk)
        except RequestFailed as exc:
            # Retries are already exhausted inside PoliteSession. Losing one
            # chunk costs a few candidate rows; it is logged, never silenced.
            logger.warning("Giving up on a %s chunk of %d ids: %s", what, len(chunk), exc)
            self.stats.drop(f"{what}_fetch_failed", len(chunk))
            return {}

    # -- value extraction --------------------------------------------------

    @staticmethod
    def _snak_value(snak: dict) -> str:
        snaktype = snak.get("snaktype")
        if snaktype == "somevalue":
            return _SOME_VALUE
        if snaktype == "novalue":
            return _NO_VALUE
        if snaktype != "value":
            return _ERROR
        value = snak.get("datavalue", {}).get("value")
        if isinstance(value, dict):
            for key in ("id", "amount", "time", "text"):
                if key in value:
                    return str(value[key])
            if "latitude" in value and "longitude" in value:
                return f"{value['latitude']},{value['longitude']}"
        return str(value)

    def _extract_property_snapshot(self, entity_data: Optional[dict], property_id: str) -> Tuple[str, str]:
        """Return ``(value, datatype)`` for ``property_id`` in a revision blob.

        Rank filtering happens here via :func:`filters.best_statement`, which
        the previous release skipped -- it always read ``claims[pid][0]`` and so
        could report a deprecated statement as the live value.
        """
        if not entity_data:
            return _ERROR, ""
        statements = entity_data.get("claims", {}).get(property_id)
        if not statements:
            return _NO_VALUE, ""
        statement = filters.best_statement(statements)
        if statement is None:
            return _NO_VALUE, ""
        snak = statement.get("mainsnak", {})
        return self._snak_value(snak), snak.get("datatype", "")

    # -- per-batch pipeline ------------------------------------------------

    def process_batch(self, changes: List[Dict]) -> List[List[str]]:
        """Turn one recentchanges batch into CSV rows."""
        self._counts["scanned"] += len(changes)

        item_changes = [c for c in changes if str(c.get("title", "")).startswith("Q")]
        self._counts["item_edits"] += len(item_changes)
        if not item_changes:
            return []

        # Only edits whose comment names a property can be diffed cheaply.
        pending = []
        for change in item_changes:
            match = _PROPERTY_IN_COMMENT.search(change.get("comment", "") or "")
            if not match:
                self.stats.drop("no property in edit comment")
                continue
            pending.append((change, match.group(1)))
        self._counts["predicate_in_comment"] += len(pending)
        if not pending:
            return []

        # Cheap deny-list pass before any entity traffic.
        survivors = []
        for change, pid in pending:
            if not filters.is_allowed_relation(property_id=pid, use_allowlist=False):
                self.stats.drop("relation on deny-list")
                continue
            survivors.append((change, pid))
        if not survivors:
            return []

        entity_map = self._fetch_entities(sorted({c["title"] for c, _ in survivors}))

        candidates = []
        for change, pid in survivors:
            qid = change["title"]
            entity = entity_map.get(qid)
            if not entity or "missing" in entity:
                self.stats.drop("entity missing or unfetchable")
                continue

            sitelinks = entity.get("sitelinks", {})
            label = entity.get("labels", {}).get("en", {}).get("value", "")
            instance_of = [
                statement.get("mainsnak", {}).get("datavalue", {}).get("value", {}).get("id")
                for statement in entity.get("claims", {}).get("P31", [])
            ]
            ok, reason = filters.is_allowed_entity(
                label=label,
                sitelinks=sitelinks,
                instance_of=[q for q in instance_of if q],
                require_enwiki=self.require_enwiki,
            )
            if not ok:
                self.stats.drop(f"entity: {reason}")
                continue

            enwiki_title = sitelinks.get("enwiki", {}).get("title", "")
            if enwiki_title.startswith(("Category:", "Template:", "Portal:", "Help:")):
                self.stats.drop("entity: enwiki title is a Wikimedia namespace page")
                continue
            self._counts["entity_passed"] += 1

            statements = entity.get("claims", {}).get(pid) or []
            datatype = ""
            if statements:
                best = filters.best_statement(statements)
                if best is not None:
                    datatype = best.get("mainsnak", {}).get("datatype", "")

            candidates.append({
                "change": change,
                "qid": qid,
                "label": label or qid,
                "pid": pid,
                "enwiki_title": enwiki_title,
                "datatype": datatype,
            })

        if not candidates:
            return []

        property_labels = self._fetch_labels(sorted({c["pid"] for c in candidates}))
        allowed = []
        for candidate in candidates:
            candidate["property_label"] = property_labels.get(candidate["pid"], candidate["pid"])
            if not filters.is_allowed_relation(
                property_id=candidate["pid"],
                property_label=candidate["property_label"],
                use_allowlist=self.use_allowlist,
            ):
                self.stats.drop("relation not in allow-list")
                continue
            allowed.append(candidate)
        self._counts["relation_passed"] += len(allowed)
        if not allowed:
            return []

        # Deleted statements are absent from the *current* entity, so their
        # datatype is unknown here and is recovered from the old revision below.
        typed = []
        for candidate in allowed:
            datatype = candidate["datatype"]
            if datatype and datatype not in ALLOWED_TYPES:
                self.stats.drop(f"datatype not allowed: {datatype}")
                continue
            typed.append(candidate)
        if not typed:
            return []

        revids: Set[str] = set()
        for candidate in typed:
            change = candidate["change"]
            if change.get("revid"):
                revids.add(str(change["revid"]))
            try:
                old_revid = int(change.get("old_revid") or 0)
            except (TypeError, ValueError):
                old_revid = 0
            if old_revid > 0:
                revids.add(str(old_revid))
        revisions = self._batch_fetch_revisions(sorted(revids))

        resolved = []
        for candidate in typed:
            outcome = self._resolve_diff(candidate, revisions)
            if outcome is None:
                continue
            resolved.append(outcome)
        self._counts["diff_resolved"] += len(resolved)

        in_scope = []
        object_ids: Set[str] = set()
        for row in resolved:
            if row["change_type"] not in self.change_types:
                self.stats.drop(f"change_type filtered out: {row['change_type']}")
                continue
            if row["new_value"].startswith("Q") and row["new_value"][1:].isdigit():
                object_ids.add(row["new_value"])
            in_scope.append(row)
        self._counts["change_type_in_scope"] += len(in_scope)
        if not in_scope:
            return []

        object_labels = self._fetch_labels(sorted(object_ids))

        rows: List[List[str]] = []
        for row in in_scope:
            new_value = row["new_value"]
            rows.append([
                row["qid"], row["label"], row["pid"], row["property_label"], row["datatype"],
                row["old_value"], new_value, object_labels.get(new_value, new_value),
                row["change_type"], row["timestamp"], row["wiki_url"],
            ])
        # recentchanges is newest-first; ordering each batch keeps the CSV
        # monotonically descending so the first row per entity is the newest.
        rows.sort(key=lambda r: r[_COL["change_timestamp"]], reverse=True)
        return rows

    def _resolve_diff(self, candidate: dict, revisions: Dict[str, Optional[dict]]) -> Optional[dict]:
        change = candidate["change"]
        pid = candidate["pid"]
        revid_new = str(change.get("revid", ""))
        try:
            old_revid = int(change.get("old_revid") or 0)
        except (TypeError, ValueError):
            old_revid = 0

        if revid_new not in revisions:
            self.stats.drop("new revision unavailable")
            return None

        new_value, new_type = self._extract_property_snapshot(revisions.get(revid_new), pid)
        if old_revid > 0:
            if str(old_revid) not in revisions:
                self.stats.drop("old revision unavailable")
                return None
            old_value, old_type = self._extract_property_snapshot(revisions.get(str(old_revid)), pid)
        else:
            old_value, old_type = _NEW_CREATED, ""

        if _ERROR in (new_value, old_value):
            self.stats.drop("value could not be parsed")
            return None
        if new_value == old_value:
            self.stats.drop("property unchanged in this revision")
            return None

        datatype = candidate["datatype"] or new_type or old_type
        if datatype not in ALLOWED_TYPES:
            self.stats.drop(f"datatype not allowed: {datatype or 'unknown'}")
            return None

        if old_value in (_NEW_CREATED, _NO_VALUE, _SOME_VALUE):
            change_type = "created"
        elif new_value in (_NO_VALUE, _SOME_VALUE):
            change_type = "deleted"
        else:
            change_type = "updated"

        return {
            "qid": candidate["qid"],
            "label": candidate["label"],
            "pid": pid,
            "property_label": candidate["property_label"],
            "datatype": datatype,
            "old_value": old_value,
            "new_value": new_value,
            "change_type": change_type,
            "timestamp": change.get("timestamp", ""),
            # Empty rather than a bare domain when --allow-missing-enwiki is in use.
            "wiki_url": ("https://en.wikipedia.org/wiki/"
                         + candidate["enwiki_title"].replace(" ", "_")) if candidate["enwiki_title"] else "",
        }

    # -- driver ------------------------------------------------------------

    def _finalise_stats(self) -> None:
        self.stats.stage("recentchanges scanned", self._counts["scanned"])
        self.stats.stage("item (ns 0) edits", self._counts["item_edits"])
        self.stats.stage("edit comment names a property", self._counts["predicate_in_comment"])
        self.stats.stage("entity filter passed", self._counts["entity_passed"])
        self.stats.stage("relation filter passed", self._counts["relation_passed"])
        self.stats.stage("value diff resolved", self._counts["diff_resolved"])
        self.stats.stage("change type in scope", self._counts["change_type_in_scope"])
        self.stats.stage("unique triples written", self._counts["written"])

    def _write_stats(self, sink: CsvSink, duration: float) -> Tuple[Path, Path]:
        payload = self.stats.to_dict()
        payload["run"] = {
            "output": str(sink.path),
            "window_start_utc": get_timestamp_str(self.start_time) if self.start_time else None,
            "window_end_utc": get_timestamp_str(self.end_time),
            "hours": self.hours,
            "rc_limit": self.rc_limit,
            "change_types": sorted(self.change_types),
            "use_allowlist": self.use_allowlist,
            "max_workers": self.max_workers,
            "max_triples": self.max_triples,
            "require_enwiki": self.require_enwiki,
            "api_endpoint": self.api_endpoint,
            "rows_written_this_run": self._counts["written"],
            "rows_in_file": len(sink.seen_entities),
            "duration_seconds": round(duration, 2),
        }
        json_path = Path(str(sink.path) + ".stats.json")
        md_path = Path(str(sink.path) + ".stats.md")
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        md_path.write_text(
            "# Extraction funnel\n\n"
            f"Window: {payload['run']['window_start_utc'] or 'unbounded'} -> "
            f"{payload['run']['window_end_utc']} (UTC)\n\n"
            + self.stats.render() + "\n",
            encoding="utf-8",
        )
        return json_path, md_path

    def run(self) -> None:
        if self.start_time is not None:
            logger.info("Window: %s -> %s UTC (%.2f h)", self.start_time, self.end_time, self.hours)
        else:
            logger.info("Window: unbounded, scanning back from %s UTC", self.end_time)
        logger.info("rclimit=%d, workers=%d, change types=%s, allow-list=%s",
                    self.rc_limit, self.max_workers,
                    ",".join(sorted(self.change_types)),
                    "on" if self.use_allowlist else "off")

        # Opened before the scan: a failure here must not cost 90 minutes.
        sink = CsvSink(self.output_file, resume=self.resume)
        logger.info("Streaming rows to %s", sink.path)

        started = time.time()
        try:
            for batch in self.fetch_recent_changes_generator():
                self.total_changes += len(batch)
                rows = self.process_batch(batch)
                if rows:
                    added = sink.write_rows(rows)
                    self._counts["written"] += added
                    for _ in range(len(rows) - added):
                        self.stats.drop("duplicate entity (older change)")
                logger.info("scanned=%d  written=%d", self.total_changes, self._counts["written"])
                if self.max_triples and self._counts["written"] >= self.max_triples:
                    logger.info("Reached --max-triples=%d, stopping scan", self.max_triples)
                    break
        except KeyboardInterrupt:
            logger.warning("Interrupted; %d rows already flushed to %s",
                           self._counts["written"], sink.path)
        finally:
            duration = time.time() - started
            self._finalise_stats()
            sink.close()
            json_path, md_path = self._write_stats(sink, duration)
            self.close_sessions()

        self.kept_triples = self._counts["written"]
        logger.info("Done in %.1fs: scanned %d changes, wrote %d triples",
                    duration, self.total_changes, self.kept_triples)
        logger.info("CSV:   %s", sink.path)
        logger.info("Stats: %s / %s", json_path, md_path)
        print(self.stats.render())


# ========== Entry point ==========

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_triple_changes.py",
        description="Extract knowledge triple changes from the Wikidata recentchanges feed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Notes:\n"
            "  * Wikidata keeps recentchanges for roughly 30 days. For older windows use\n"
            "    scripts/extract_dump_delta.py, which diffs two JSON dumps instead.\n"
            "  * The CSV is written incrementally, so an interrupted run is still usable\n"
            "    and can be continued with --resume.\n"
        ),
    )
    parser.add_argument("--hours", type=float, default=None,
                        help="Look-back window in hours (default: 2.0, or unbounded when "
                             "--max-triples is given).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path; the parent directory is created for you "
                             "(default: outputs/extracted_triples/triple_changes_<UTC timestamp>.csv).")
    parser.add_argument("--max-triples", type=int, default=None,
                        help="Stop once this many unique-entity rows have been written by this run "
                             "(rows already in the file under --resume do not count towards it).")
    parser.add_argument("--rc-limit", type=int, default=200,
                        help="recentchanges page size (default: 200). The old hardcoded 500 is "
                             "what triggered internal_api_error_DBQueryTimeoutError; lower this "
                             "further if you still see timeouts.")
    parser.add_argument("--change-type", type=str, default=DEFAULT_CHANGE_TYPES,
                        help="Comma-separated subset of created,updated,deleted -- or 'all' "
                             f"(default: {DEFAULT_CHANGE_TYPES}, the paper's delta definition).")
    parser.add_argument("--no-allowlist", action="store_true",
                        help="Disable the curated 198-relation allow-list and keep every relation "
                             "that survives the deny-list. Widens recall, lowers question quality.")
    parser.add_argument("--allow-missing-enwiki", action="store_true",
                        help="Keep subjects with no English Wikipedia sitelink (off by default).")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Threads used for entity/revision batch fetching (default: 10). "
                             "Use 1 for a strictly serial, maximally polite run.")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help="HTTP attempts per request before giving up (default: 5).")
    parser.add_argument("--min-interval", type=float, default=0.0,
                        help="Minimum seconds between requests on a single connection (default: 0).")
    parser.add_argument("--api-endpoint", type=str, default=None,
                        help="Wikidata Action API endpoint (default: $WIKIDATA_API or "
                             f"{config.DEFAULT_WIKIDATA_API}).")
    parser.add_argument("--resume", action="store_true",
                        help="Append to an existing --output instead of overwriting it, skipping "
                             "entities already present.")
    parser.add_argument("--force", action="store_true",
                        help="Proceed even when --hours exceeds the ~30-day recentchanges retention.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate everything before a single request goes out.
    if args.hours is not None and args.hours <= 0:
        raise SystemExit("--hours must be positive")
    if args.max_triples is not None and args.max_triples <= 0:
        raise SystemExit("--max-triples must be positive")
    if args.rc_limit > 500:
        raise SystemExit("--rc-limit above 500 is rejected by the Wikidata API for anonymous clients")

    try:
        change_types = parse_change_types(args.change_type)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc))

    hours = args.hours
    if hours is None:
        hours = None if args.max_triples else 2.0

    if hours is not None and hours > RECENTCHANGES_RETENTION_HOURS:
        message = (
            f"--hours {hours:g} asks for {hours / 24:.1f} days, but Wikidata prunes "
            f"recentchanges after about {RECENTCHANGES_RETENTION_HOURS // 24} days.\n"
            "  Anything older simply returns zero changes -- silently, with no API error.\n"
            "  Use scripts/extract_dump_delta.py to diff two JSON dumps for older periods,\n"
            "  or pass --force to run this window anyway."
        )
        if not args.force:
            raise SystemExit("Refusing to run: " + message)
        logger.warning(message)
    elif hours is None:
        logger.warning("Unbounded window requested: the scan will stop at the end of the "
                       "recentchanges table (~%d days back), not at the beginning of Wikidata.",
                       RECENTCHANGES_RETENTION_HOURS // 24)

    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = str(EXTRACTED_TRIPLES_DIR / f"triple_changes_{stamp}.csv")
    else:
        output_file = args.output

    extractor = TripleChangeExtractor(
        hours=hours,
        output_file=output_file,
        max_triples=args.max_triples,
        rc_limit=args.rc_limit,
        change_types=change_types,
        use_allowlist=not args.no_allowlist,
        max_workers=args.max_workers,
        resume=args.resume,
        api_endpoint=args.api_endpoint,
        max_attempts=args.max_attempts,
        min_interval=args.min_interval,
        require_enwiki=not args.allow_missing_enwiki,
    )
    try:
        extractor.run()
    except RequestFailed as exc:
        # Bounded failure instead of the old infinite retry loop.
        raise SystemExit(f"Wikidata API unreachable: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
