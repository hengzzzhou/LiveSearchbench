"""Wikidata recent changes RDF diff pipeline per task.md."""
from __future__ import annotations

import csv
import functools
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

import requests
from rdflib import Graph

BASE_API = "https://www.wikidata.org/w/api.php"
UA = "TriplesDiffBot/0.1 (contact: you@example.com)"

Triple = Tuple[str, str, str]
DiffResult = Dict[str, object]
CsvRow = Dict[str, str]

CSV_COLUMNS = [
    "subject_id",
    "subject_label",
    "predicate_id",
    "predicate_label",
    "object_id",
    "object_label",
    "add or change",
]

LABEL_LANGS = ("en",)
WIKIDATA_ID_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9]*\d+$")
LABEL_CACHE: Dict[str, str] = {}
LABEL_CACHE_LOCK = threading.Lock()


def rc_window(days: int = 30) -> Tuple[str, str]:
    """Return rcstart(now) and rcend(now - days) ISO8601 timestamps."""
    now = datetime.now(timezone.utc)
    start = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return start, end


def retry(n: int = 3, base: int = 2):
    """Simple retry decorator with exponential backoff."""

    def wrap(fn):
        @functools.wraps(fn)
        def run(*args, **kwargs):
            delay = 1.0
            for attempt in range(n):
                try:
                    return fn(*args, **kwargs)
                except Exception:
                    if attempt == n - 1:
                        raise
                    time.sleep(delay + random.random())
                    delay *= base

        return run

    return wrap


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def extract_wikidata_id(value: str) -> Optional[str]:
    """Best-effort extraction of Wikidata entity/property ID from an IRI."""
    if not is_url(value):
        return None
    if "/entity/statement/" in value:
        tail = value.split("/entity/statement/")[-1]
        base = tail.split("-")[0]
        if WIKIDATA_ID_PATTERN.match(base):
            return base
    for token in (
        "/entity/",
        "/wiki/Special:EntityData/",
        "/prop/direct/",
        "/prop/statement/",
        "/prop/qualifier/",
        "/prop/reference/",
        "/prop/numeric-id/",
        "/prop/",
    ):
        if token in value:
            tail = value.split(token)[-1]
            tail = tail.split("/")[0]
            tail = tail.split("#")[0]
            if WIKIDATA_ID_PATTERN.match(tail):
                return tail
    tail = value.rsplit("/", 1)[-1]
    tail = tail.split("#")[0]
    if WIKIDATA_ID_PATTERN.match(tail):
        return tail
    return None


def normalize_subject_id(value: str) -> str:
    if "/entity/statement/" in value:
        return value.split("/entity/statement/")[-1]
    identifier = extract_wikidata_id(value)
    return identifier or value


def normalize_predicate_id(value: str) -> str:
    identifier = extract_wikidata_id(value)
    return identifier or value


def normalize_object(value: str) -> Tuple[str, bool, str]:
    """Return (object_id, is_literal, raw_value)."""
    if is_url(value):
        if "/entity/statement/" in value:
            identifier = value.split("/entity/statement/")[-1]
            return (identifier, False, value)
        identifier = extract_wikidata_id(value)
        return (identifier or value, False, value)
    return (value, True, value)


def is_entity_id(value: str) -> bool:
    return bool(value) and value.startswith("Q") and value[1:].isdigit()


def is_property_id(value: str) -> bool:
    return bool(value) and value.startswith("P") and value[1:].isdigit()


@retry()
def fetch_rc_page(
    rcstart: str,
    rcend: str,
    rccontinue: Optional[str] = None,
    limit: int = 500,
) -> Dict[str, object]:
    """Fetch one page of recentchanges results."""
    params = dict(
        action="query",
        list="recentchanges",
        format="json",
        rcnamespace=0,
        rctype="edit|new",
        rcprop="ids|title|timestamp|type",
        rcstart=rcstart,
        rcend=rcend,
        rcdir="older",
        rclimit=str(limit),
        formatversion="2",
        maxlag="5",
    )
    if rccontinue:
        params["rccontinue"] = rccontinue
    response = requests.get(
        BASE_API,
        params=params,
        headers={"User-Agent": UA},
        timeout=60,
    )
    if response.status_code == 429:
        wait = int(response.headers.get("Retry-After", "5"))
        time.sleep(wait)
        response = requests.get(
            BASE_API,
            params=params,
            headers={"User-Agent": UA},
            timeout=60,
        )
    response.raise_for_status()
    return response.json()


def _time_slices(days: int, chunk_days: int = 1) -> Iterator[Tuple[str, str]]:
    now = datetime.now(timezone.utc)
    for offset in range(0, days, chunk_days):
        start_dt = now - timedelta(days=offset)
        end_dt = now - timedelta(days=min(days, offset + chunk_days))
        yield (
            start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            end_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        )


def iter_recent_changes(
    days: int = 30,
    max_pages: Optional[int] = None,
    chunk_days: int = 1,
) -> Iterator[Dict[str, object]]:
    """Yield recent change items page by page (optionally sliced by day)."""
    seen_rcids: Set[int] = set()
    pages_emitted = 0
    for rcstart, rcend in _time_slices(days, chunk_days=chunk_days):
        print(f"[INFO] Fetching window {rcstart} -> {rcend}")
        rccont = None
        while True:
            try:
                data = fetch_rc_page(rcstart, rcend, rccont)
            except Exception as exc:
                print(f"WARN: fetch_rc_page({rcstart}->{rcend}) failed: {exc}")
                break
            page = data.get("query", {}).get("recentchanges", [])
            if not page:
                break
            for entry in page:
                rcid = entry.get("rcid")
                if rcid and rcid in seen_rcids:
                    continue
                if rcid:
                    seen_rcids.add(rcid)
                yield entry
            pages_emitted += 1
            print(f"[INFO]   page size={len(page)} total_pages={pages_emitted}")
            if max_pages is not None and pages_emitted >= max_pages:
                return
            rccont = data.get("continue", {}).get("rccontinue")
            if not rccont:
                break


def fetch_recent_changes(days: int = 30, max_pages: Optional[int] = None) -> List[Dict[str, object]]:
    """Collect recent changes into a list (mainly for diagnostics)."""
    return list(iter_recent_changes(days=days, max_pages=max_pages))


def simplify_changes(changes: Iterable[Dict[str, object]]) -> Iterator[Dict[str, object]]:
    """Extract entity/revision identifiers from raw change entries."""
    for c in changes:
        yield dict(
            entity_id=c.get("title"),
            new_id=c.get("revid", 0),
            old_id=c.get("old_revid", 0),
            timestamp=c.get("timestamp"),
            type=c.get("type"),
        )


@retry()
def entity_ttl(entity_id: str, revision_id: Optional[int] = None) -> str:
    """Fetch turtle document for a given entity revision."""
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.ttl?flavor=dump"
    if revision_id:
        url += f"&revision={revision_id}"
    response = requests.get(
        url,
        headers={"User-Agent": UA, "Accept-Encoding": "gzip"},
        timeout=60,
    )
    if response.status_code == 429:
        wait = int(response.headers.get("Retry-After", "5"))
        time.sleep(wait)
        response = requests.get(
            url,
            headers={"User-Agent": UA, "Accept-Encoding": "gzip"},
            timeout=60,
        )
    response.raise_for_status()
    return response.text


@retry()
def _fetch_label_chunk(ids: List[str]) -> Dict[str, Dict[str, object]]:
    params = dict(
        action="wbgetentities",
        ids="|".join(ids),
        props="labels",
        languages="|".join(LABEL_LANGS),
        format="json",
    )
    response = requests.get(
        BASE_API,
        params=params,
        headers={"User-Agent": UA},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("entities", {})


def resolve_labels(ids: Set[str]) -> None:
    if not ids:
        return
    with LABEL_CACHE_LOCK:
        missing = [identifier for identifier in ids if identifier not in LABEL_CACHE]
    if not missing:
        return
    chunk_size = 50
    for idx in range(0, len(missing), chunk_size):
        chunk = missing[idx : idx + chunk_size]
        entities = _fetch_label_chunk(chunk)
        for identifier in chunk:
            entry = entities.get(identifier)
            if not entry or "missing" in entry:
                with LABEL_CACHE_LOCK:
                    LABEL_CACHE.setdefault(identifier, identifier)
                continue
            labels = entry.get("labels", {})
            label_value: Optional[str] = None
            for lang in LABEL_LANGS:
                if lang in labels:
                    label_value = labels[lang]["value"]
                    break
            if label_value is None and labels:
                label_value = next(iter(labels.values()))["value"]
            with LABEL_CACHE_LOCK:
                LABEL_CACHE[identifier] = label_value or identifier


def get_label(identifier: str, fallback: str) -> str:
    if not identifier:
        return fallback
    with LABEL_CACHE_LOCK:
        return LABEL_CACHE.get(identifier, fallback)


def parse_triples(ttl_text: str) -> Set[Triple]:
    """Parse a TTL string into a set of RDF triples."""
    graph = Graph()
    graph.parse(data=ttl_text, format="turtle")
    return {(
        str(subject),
        str(predicate),
        str(obj),
    ) for subject, predicate, obj in graph}


def diff_triples(old_set: Set[Triple], new_set: Set[Triple]) -> Dict[str, List[Triple]]:
    """Compute added/removed triples between two sets."""
    added = sorted(new_set - old_set)
    removed = sorted(old_set - new_set)
    return {"added": added, "removed": removed}


def diff_for_change(entity_id: str, new_id: int, old_id: int) -> DiffResult:
    """Compute triple diff for a change entry."""
    new_set: Set[Triple] = set()
    old_set: Set[Triple] = set()
    if new_id:
        new_set = parse_triples(entity_ttl(entity_id, new_id))
    if old_id:
        old_set = parse_triples(entity_ttl(entity_id, old_id))
    diff = diff_triples(old_set, new_set)
    result: DiffResult = dict(
        entity_id=entity_id,
        new_id=new_id,
        old_id=old_id,
        **diff,
    )
    return result


def normalize_for_row(subject: str, predicate: str, obj: str) -> Optional[Tuple[str, str, str]]:
    subject_id = normalize_subject_id(subject)
    if not is_entity_id(subject_id):
        return None
    predicate_id = normalize_predicate_id(predicate)
    if not is_property_id(predicate_id):
        return None
    object_id, is_literal, _ = normalize_object(obj)
    if is_literal or not is_entity_id(object_id):
        return None
    return subject_id, predicate_id, object_id


def collect_ids_from_triples(triples: Iterable[Triple]) -> Set[str]:
    identifiers: Set[str] = set()
    for subject, predicate, obj in triples:
        normalized = normalize_for_row(subject, predicate, obj)
        if not normalized:
            continue
        subject_id, predicate_id, object_id = normalized
        identifiers.update((subject_id, predicate_id, object_id))
    return identifiers


def triple_to_row(subject: str, predicate: str, obj: str, change_type: str) -> Optional[CsvRow]:
    normalized = normalize_for_row(subject, predicate, obj)
    if not normalized:
        return None
    subject_id, predicate_id, object_id = normalized

    subject_label = get_label(subject_id, subject_id)
    predicate_label = get_label(predicate_id, predicate_id)
    object_label = get_label(object_id, object_id)

    row: CsvRow = {
        "subject_id": subject_id,
        "subject_label": subject_label,
        "predicate_id": predicate_id,
        "predicate_label": predicate_label,
        "object_id": object_id,
        "object_label": object_label,
        "add or change": change_type,
    }
    return row


def rows_from_diff(diff: DiffResult) -> Iterator[CsvRow]:
    added = diff.get("added", [])
    removed = diff.get("removed", [])
    ids_needed = collect_ids_from_triples(added)
    if ids_needed:
        resolve_labels(ids_needed)
    removed_set: Set[Tuple[str, str, str]] = set()
    for subject, predicate, obj in removed:
        normalized = normalize_for_row(subject, predicate, obj)
        if not normalized:
            continue
        removed_set.add(normalized)
    for subject, predicate, obj in added:
        normalized = normalize_for_row(subject, predicate, obj)
        if not normalized:
            continue
        change_type = "changed" if normalized in removed_set else "added"
        row = triple_to_row(subject, predicate, obj, change_type)
        if row:
            yield row


def process_change(row: Dict[str, object], sleep_sec: float) -> List[CsvRow]:
    entity_id = row.get("entity_id")
    try:
        diff = diff_for_change(row["entity_id"], row["new_id"], row["old_id"])
        rows = list(rows_from_diff(diff))
        return rows
    except Exception as exc:
        print(f"WARN: {entity_id} -> {exc}")
        return []
    finally:
        if sleep_sec:
            time.sleep(sleep_sec)


def stream_changes(
    days: int = 30,
    limit: Optional[int] = None,
    sleep_sec: float = 0.5,
    chunk_days: int = 1,
    workers: int = 4,
    max_pending: int = 32,
    progress_every: int = 50,
) -> Iterator[CsvRow]:
    """Yield flattened triple rows for recent changes using a thread pool."""
    processed = 0
    scheduled = 0
    pending: Set[Future] = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for row in simplify_changes(
            iter_recent_changes(days=days, chunk_days=chunk_days)
        ):
            if limit is not None and scheduled >= limit:
                break
            future = executor.submit(process_change, row, sleep_sec)
            pending.add(future)
            scheduled += 1
            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    rows = fut.result()
                    processed += 1
                    if processed % progress_every == 0:
                        print(f"[INFO] Processed entities: {processed}")
                    for item in rows:
                        yield item
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                rows = fut.result()
                processed += 1
                if processed % progress_every == 0:
                    print(f"[INFO] Processed entities: {processed}")
                for item in rows:
                    yield item


def write_results(
    path: str,
    items: Iterable[CsvRow],
) -> None:
    """Write triple rows to CSV."""
    count = 0
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for item in items:
            writer.writerow(item)
            count += 1
    print(f"Saved {count} rows to {path}")


def run(
    days: int = 30,
    limit: Optional[int] = None,
    sleep_sec: float = 0.5,
    out_path: str = "results.csv",
    chunk_days: int = 1,
    workers: int = 4,
    max_pending: int = 32,
    progress_every: int = 50,
) -> None:
    """Entry point combining pagination, diffing and output."""
    print(
        f"[INFO] run start days={days} limit={limit or 'ALL'} sleep_sec={sleep_sec} "
        f"workers={workers} chunk_days={chunk_days}"
    )
    start_time = time.time()
    rows_iter = stream_changes(
        days=days,
        limit=limit,
        sleep_sec=sleep_sec,
        chunk_days=chunk_days,
        workers=workers,
        max_pending=max_pending,
        progress_every=progress_every,
    )
    write_results(out_path, rows_iter)
    elapsed = time.time() - start_time
    print(f"[INFO] run finished Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    run(days=1, limit=5, sleep_sec=0.1, workers=4, chunk_days=1)
