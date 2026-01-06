"""
Step 0: extract knowledge triple changes from Wikidata.

Features:
1. Pull recent Wikidata edits
2. Keep entities with English Wikipedia pages
3. Capture property value changes (create/update/delete)
4. Emit a CSV for downstream use

Example:
    python extract_triple_changes.py --hours 2.0
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import requests
import csv
import time
import json
import logging
import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import List, Dict
import re


# ========== Configuration ==========

# Wikidata API
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_USER_AGENT = "LiveSearchBench/1.0"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXTRACTED_TRIPLES_DIR = OUTPUTS_DIR / "extracted_triples"

# Create directories if missing
EXTRACTED_TRIPLES_DIR.mkdir(parents=True, exist_ok=True)


# ========== Logging ==========

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("triple_extractor")


# ========== Defaults ==========

BASE_API = WIKIDATA_API
USER_AGENT = WIKIDATA_USER_AGENT
MAX_WORKERS = 10  # Parallel workers
BATCH_SIZE = 50   # Batch request size

# Allowed property datatypes
ALLOWED_TYPES = {
    "time",             # Time
    "quantity",         # Quantity
    "wikibase-item",    # Wikibase item
    "globe-coordinate"  # Coordinates
}


# ========== Helpers ==========

def get_timestamp_str(dt: datetime) -> str:
    """Format datetime as a Wikidata timestamp string."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ========== Main class ==========

class TripleChangeExtractor:
    """Extract knowledge triple changes from Wikidata."""

    def __init__(self, hours: float = None, output_file: str = "triple_changes.csv", max_triples: int = None):
        """
        Init extractor.

        Args:
            hours: Time window in hours (None means unbounded).
            output_file: Output CSV file path.
            max_triples: Max triples to save (None means unbounded).
        """
        # Compute UTC time range
        self.end_time = datetime.now(timezone.utc).replace(tzinfo=None)
        if hours is not None:
            self.start_time = self.end_time - timedelta(hours=hours)
        else:
            self.start_time = None
        self.output_file = output_file
        self.max_triples = max_triples
        self.hours = hours

        # Counters
        self.total_changes = 0
        self.kept_triples = 0

    def fetch_recent_changes_generator(self):
        """
        Yield batches of recent change records.
        """
        rcstart = get_timestamp_str(self.end_time)

        params = {
            "action": "query",
            "list": "recentchanges",
            "format": "json",
            "rcnamespace": 0,
            "rcprop": "ids|title|timestamp|comment",
            "rctype": "edit|new",
            "rclimit": 500,
            "rcstart": rcstart,
            "formatversion": "2"
        }

        # Only set rcend when a time window is provided
        if self.start_time is not None:
            params["rcend"] = get_timestamp_str(self.start_time)

        while True:
            try:
                resp = requests.get(BASE_API, params=params, headers={"User-Agent": USER_AGENT})
                data = resp.json()

                if "error" in data:
                    logger.warning(f"API error: {data['error']}")
                    time.sleep(5)
                    continue

                changes = data.get("query", {}).get("recentchanges", [])
                if not changes:
                    break

                yield changes

                if "continue" in data:
                    params.update(data["continue"])
                else:
                    break

            except Exception as e:
                logger.error(f"Network error: {e}")
                time.sleep(5)

    def _fetch_property_labels(self, property_ids: List[str]) -> Dict[str, str]:
        """
        Fetch property labels in batches.
        """
        if not property_ids:
            return {}

        labels = {}
        chunks = [property_ids[i:i + BATCH_SIZE] for i in range(0, len(property_ids), BATCH_SIZE)]

        for chunk in chunks:
            try:
                params = {
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "format": "json",
                    "props": "labels",
                    "languages": "en"
                }
                resp = requests.get(BASE_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
                data = resp.json()

                entities = data.get("entities", {})
                for pid, entity_data in entities.items():
                    if "labels" in entity_data and "en" in entity_data["labels"]:
                        labels[pid] = entity_data["labels"]["en"]["value"]
                    else:
                        labels[pid] = pid

            except Exception as e:
                logger.warning(f"Failed to fetch property labels: {e}")
                for pid in chunk:
                    if pid not in labels:
                        labels[pid] = pid

        return labels

    def _fetch_entity_labels(self, entity_ids: List[str]) -> Dict[str, str]:
        """
        Fetch entity labels in batches.
        """
        if not entity_ids:
            return {}

        labels = {}
        chunks = [entity_ids[i:i + BATCH_SIZE] for i in range(0, len(entity_ids), BATCH_SIZE)]

        for chunk in chunks:
            try:
                params = {
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "format": "json",
                    "props": "labels",
                    "languages": "en"
                }
                resp = requests.get(BASE_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
                data = resp.json()

                entities = data.get("entities", {})
                for eid, entity_data in entities.items():
                    if "labels" in entity_data and "en" in entity_data["labels"]:
                        labels[eid] = entity_data["labels"]["en"]["value"]
                    else:
                        labels[eid] = eid

            except Exception as e:
                logger.warning(f"Failed to fetch entity labels: {e}")
                for eid in chunk:
                    if eid not in labels:
                        labels[eid] = eid

        return labels

    def process_batch_filters(self, changes: List[Dict]):
        """
        Process a batch of change records and return valid triple rows.
        """
        # 1. Extract unique QIDs
        unique_qids = list({c["title"] for c in changes if c["title"].startswith("Q")})
        if not unique_qids:
            return []

        # 2. Fetch entity details in batches
        entity_map = {}
        chunks = [unique_qids[i:i + BATCH_SIZE] for i in range(0, len(unique_qids), BATCH_SIZE)]

        def fetch_chunk(chunk_ids):
            """Get a chunk of entity data."""
            params = {
                "action": "wbgetentities",
                "ids": "|".join(chunk_ids),
                "format": "json",
                "props": "sitelinks|claims|labels",
            }
            try:
                r = requests.get(BASE_API, params=params, headers={"User-Agent": USER_AGENT})
                return r.json().get("entities", {})
            except:
                return {}

        for chunk in chunks:
            entity_map.update(fetch_chunk(chunk))

        # 3. Filter candidates
        candidates = []
        for c in changes:
            qid = c["title"]

            if qid not in entity_map or "missing" in entity_map[qid]:
                continue

            ent = entity_map[qid]

            # Require an English Wikipedia link
            if "enwiki" not in ent.get("sitelinks", {}):
                continue

            # Skip Wikipedia categories
            enwiki_title = ent.get("sitelinks", {}).get("enwiki", {}).get("title", "")
            if enwiki_title.startswith("Category:"):
                continue

            # Extract property ID from comment
            match = re.search(r"\[\[Property:(P\d+)\]\]", c.get("comment", ""))
            if not match:
                continue

            pid = match.group(1)
            claims = ent.get("claims", {})

            # Determine property type
            p_type = "unknown"
            if pid in claims and claims[pid]:
                p_type = claims[pid][0].get("mainsnak", {}).get("datatype", "unknown")

            # Property type must be allowed
            if p_type not in ALLOWED_TYPES:
                continue

            candidates.append({
                "change": c,
                "ent": ent,
                "pid": pid,
                "p_label": pid,
                "p_type": p_type
            })

        if not candidates:
            return []

        # 4. Fetch property labels in bulk
        unique_pids = list({c["pid"] for c in candidates})
        property_labels = self._fetch_property_labels(unique_pids)

        for cand in candidates:
            cand["p_label"] = property_labels.get(cand["pid"], cand["pid"])

        # 5. Fetch revisions in bulk
        all_revids = set()
        for cand in candidates:
            c = cand["change"]
            if c.get("revid"):
                all_revids.add(str(c["revid"]))
            if c.get("old_revid") and int(c.get("old_revid", 0)) > 0:
                all_revids.add(str(c["old_revid"]))

        revision_cache = self._batch_fetch_revisions(list(all_revids))

        # 6. Collect object IDs needing labels
        object_ids_to_fetch = set()
        for cand in candidates:
            c = cand["change"]
            qid = c["title"]
            pid = cand["pid"]

            revid_new = str(c.get("revid", ""))
            if revid_new in revision_cache:
                new_val = self._extract_property_value(revision_cache.get(revid_new), pid)
                # Fetch labels for wikibase item values
                if new_val and isinstance(new_val, str) and new_val.startswith("Q"):
                    object_ids_to_fetch.add(new_val)

        # Fetch object labels
        object_labels = self._fetch_entity_labels(list(object_ids_to_fetch))

        # 7. Parse diffs concurrently
        def resolve_diff_cached(item):
            """Resolve diffs using cached revisions."""
            c = item["change"]
            qid = c["title"]
            pid = item["pid"]

            revid_new = str(c.get("revid", ""))
            revid_old = str(c.get("old_revid", ""))

            try:
                # Pull new and old values
                new_val = self._extract_property_value(revision_cache.get(revid_new), pid)

                if not revid_old or int(c.get("old_revid", 0)) == 0:
                    old_val = "NEW_CREATED"
                else:
                    old_val = self._extract_property_value(revision_cache.get(revid_old), pid)

                # Check for change
                if new_val != old_val and new_val != "ERROR" and old_val != "ERROR":
                    enwiki_title = item["ent"]["sitelinks"]["enwiki"]["title"]
                    wiki_url = f"https://en.wikipedia.org/wiki/{enwiki_title.replace(' ', '_')}"
                    entity_label = item["ent"].get("labels", {}).get("en", {}).get("value", qid)

                    # Determine change type
                    if old_val in ("NEW_CREATED", "NO_VALUE", "SOME_VALUE"):
                        change_type = "created"
                    elif new_val in ("NO_VALUE", "SOME_VALUE"):
                        change_type = "deleted"
                    else:
                        change_type = "updated"

                    # Map new_value to its label when needed
                    new_val_label = new_val
                    if isinstance(new_val, str) and new_val.startswith("Q"):
                        new_val_label = object_labels.get(new_val, new_val)

                    return [
                        qid, entity_label, pid, item["p_label"], item["p_type"],
                        old_val, new_val, new_val_label, change_type,
                        c["timestamp"],
                        wiki_url
                    ]
            except Exception:
                pass

            return None

        # Concurrent processing
        rows_to_save = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(resolve_diff_cached, cand) for cand in candidates]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    rows_to_save.append(res)

        return rows_to_save

    def _batch_fetch_revisions(self, revids: List[str]) -> Dict[str, dict]:
        """
        Fetch revision contents in batches.
        """
        if not revids:
            return {}

        revision_cache = {}
        chunks = [revids[i:i + 50] for i in range(0, len(revids), 50)]

        def fetch_chunk(chunk_ids):
            """Fetch a batch of revisions."""
            try:
                params = {
                    "action": "query",
                    "prop": "revisions",
                    "revids": "|".join(chunk_ids),
                    "rvprop": "ids|content",
                    "rvslots": "main",
                    "format": "json",
                    "formatversion": "2"
                }
                resp = requests.get(BASE_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
                data = resp.json()

                results = {}
                pages = data.get("query", {}).get("pages", [])
                for page in pages:
                    for rev in page.get("revisions", []):
                        revid = str(rev.get("revid", ""))
                        content = rev.get("slots", {}).get("main", {}).get("content")
                        if content:
                            try:
                                results[revid] = json.loads(content) if isinstance(content, str) else content
                            except:
                                results[revid] = None
                return results
            except Exception as e:
                logger.warning(f"Failed to fetch revisions: {e}")
                return {}

        # Parallel fetch
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(futures):
                revision_cache.update(future.result())

        return revision_cache

    def _extract_property_value(self, entity_data: dict, property_id: str) -> str:
        """
        Extract a property value from entity data.
        """
        if not entity_data:
            return "ERROR"

        claims = entity_data.get("claims", {})
        if property_id not in claims:
            return "NO_VALUE"

        try:
            snak = claims[property_id][0].get("mainsnak", {})
            snaktype = snak.get("snaktype")

            if snaktype == "value":
                datavalue = snak.get("datavalue", {})
                val = datavalue.get("value")

                # Handle datatypes
                if isinstance(val, dict):
                    if "id" in val:
                        return val["id"]
                    if "amount" in val:
                        return val["amount"]
                    if "time" in val:
                        return val["time"]
                    if "latitude" in val:
                        return f"{val['latitude']},{val['longitude']}"
                    if "text" in val:
                        return val["text"]

                return str(val)

            elif snaktype == "somevalue":
                return "SOME_VALUE"
            elif snaktype == "novalue":
                return "NO_VALUE"

        except:
            pass

        return "ERROR"

    def run(self):
        """Execute extraction workflow."""
        if self.start_time is not None:
            logger.info(f"Fetching changes from {self.start_time} to {self.end_time} (UTC)")
            logger.info(f"Time window: {(self.end_time - self.start_time).total_seconds() / 3600:.2f} hours")
        else:
            logger.info(f"Fetching changes from earliest to {self.end_time} (UTC)")
            logger.info("Time window: unlimited")
        if self.max_triples:
            logger.info(f"Target triple count: {self.max_triples}")

        # Deduplicate by entity_id, keeping the latest change
        unique_entities = {}
        total_before_dedup = 0
        start_t = time.time()

        # Process batches
        for batch in self.fetch_recent_changes_generator():
            self.total_changes += len(batch)

            valid_rows = self.process_batch_filters(batch)
            if valid_rows:
                total_before_dedup += len(valid_rows)
                for row in valid_rows:
                    entity_id = row[0]
                    timestamp = row[9]

                    # Keep newest change per entity
                    if entity_id not in unique_entities or timestamp > unique_entities[entity_id][9]:
                        unique_entities[entity_id] = row

                print(f"Scanned {self.total_changes} changes | Found {len(unique_entities)} unique entities...", end="\r")

                # Early exit when hitting target count
                if self.max_triples and len(unique_entities) >= self.max_triples:
                    logger.info(f"\nReached target {self.max_triples}, stopping scan")
                    break

        logger.info(f"\nBefore dedup: {total_before_dedup} changes")
        logger.info(f"After dedup: {len(unique_entities)} unique entities")

        # Sort by timestamp (index 9)
        sorted_rows = sorted(unique_entities.values(), key=lambda x: x[9], reverse=True)

        # Apply limit if requested
        if self.max_triples:
            sorted_rows = sorted_rows[:self.max_triples]
            logger.info(f"Limited output: {len(sorted_rows)} triples")

        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "entity_id", "entity_label", "property_id", "property_label", "property_type",
                "old_value", "new_value", "new_value_label", "change_type", "change_timestamp", "wiki_url"
            ])
            writer.writerows(sorted_rows)

        self.kept_triples = len(sorted_rows)
        duration = time.time() - start_t
        logger.info(f"Done! Scanned {self.total_changes} changes in {duration:.1f}s")
        logger.info(f"Saved {self.kept_triples} triples to {self.output_file}")


# ========== Entry point ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract knowledge triple changes from Wikidata")
    parser.add_argument("--hours", type=float, default=None,
                        help="Time window in hours (default: unlimited when --max-triples is set, otherwise 2.0)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (default: outputs/extracted_triples/triple_changes_<timestamp>.csv)")
    parser.add_argument("--max-triples", type=int, default=None,
                        help="Max triples to output (default: unlimited, constrained by time window)")
    args = parser.parse_args()

    # Auto-configure the time window
    if args.hours is None:
        if args.max_triples:
            # No time limit when only max_triples is set
            hours = None
        else:
            # Default window is 2 hours
            hours = 2.0
    else:
        hours = args.hours

    # Default output path
    if args.output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = str(EXTRACTED_TRIPLES_DIR / f"triple_changes_{timestamp}.csv")
    else:
        output_file = args.output

    # Startup info
    print("\n" + "=" * 60)
    print("Wikidata triple change extractor")
    print("=" * 60)
    if hours is not None:
        print(f"Time window: {hours} hours (UTC)")
    else:
        print("Time window: unlimited (scan until target count is reached)")
    if args.max_triples:
        print(f"Target count: {args.max_triples} triples")
    print(f"Output file: {output_file}")
    print("=" * 60 + "\n")

    # Run extractor
    extractor = TripleChangeExtractor(hours=hours, output_file=output_file, max_triples=args.max_triples)
    extractor.run()
