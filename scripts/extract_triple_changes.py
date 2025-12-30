"""
步骤 0：从 Wikidata 提取知识三元组变化

功能：
1. 获取 Wikidata 最近的编辑记录
2. 过滤出符合条件的实体（有英文维基百科页面）
3. 提取属性值的变化（新增、更新、删除）
4. 输出为 CSV 文件供后续使用

运行示例：
    python 0_extract_triple_changes.py --hours 2.0
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
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


# ========== 配置 ==========

# Wikidata API 配置
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_USER_AGENT = "LiveSearchBench/1.0"

# 输出目录配置
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EXTRACTED_TRIPLES_DIR = OUTPUTS_DIR / "extracted_triples"

# 创建必要的目录
EXTRACTED_TRIPLES_DIR.mkdir(parents=True, exist_ok=True)


# ========== 日志配置 ==========

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("triple_extractor")


# ========== 全局配置 ==========

BASE_API = WIKIDATA_API
USER_AGENT = WIKIDATA_USER_AGENT
MAX_WORKERS = 10  # 并行线程数
BATCH_SIZE = 50   # 批量请求大小

# 允许的属性数据类型
ALLOWED_TYPES = {
    "time",             # 时间
    "quantity",         # 数量
    "wikibase-item",    # 维基项
    "globe-coordinate"  # 地理坐标
}


# ========== 辅助函数 ==========

def get_timestamp_str(dt: datetime) -> str:
    """将 datetime 转为 Wikidata API 时间戳格式"""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ========== 主类 ==========

class TripleChangeExtractor:
    """从 Wikidata 提取知识三元组变化"""

    def __init__(self, hours: float = None, output_file: str = "triple_changes.csv", max_triples: int = None):
        """
        初始化提取器

        参数:
            hours: 扫描时间窗口（小时，None 表示不限制时间）
            output_file: 输出 CSV 文件路径
            max_triples: 最大输出三元组数量（None 表示不限制）
        """
        # 计算时间范围（UTC）
        self.end_time = datetime.now(timezone.utc).replace(tzinfo=None)
        if hours is not None:
            self.start_time = self.end_time - timedelta(hours=hours)
        else:
            # 不限制时间，设置为一个很久之前的时间
            self.start_time = None
        self.output_file = output_file
        self.max_triples = max_triples
        self.hours = hours

        # 统计计数器
        self.total_changes = 0
        self.kept_triples = 0

    def fetch_recent_changes_generator(self):
        """
        生成器：分批获取最近更改记录

        Yields:
            list: 更改记录列表
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

        # 只有当指定了时间窗口时才设置 rcend
        if self.start_time is not None:
            params["rcend"] = get_timestamp_str(self.start_time)

        while True:
            try:
                resp = requests.get(BASE_API, params=params, headers={"User-Agent": USER_AGENT})
                data = resp.json()

                if "error" in data:
                    logger.warning(f"API 错误: {data['error']}")
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
                logger.error(f"网络错误: {e}")
                time.sleep(5)

    def _fetch_property_labels(self, property_ids: List[str]) -> Dict[str, str]:
        """
        批量获取属性标签

        参数:
            property_ids: 属性 ID 列表

        返回:
            dict: {property_id: label} 映射
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
                logger.warning(f"获取属性标签失败: {e}")
                for pid in chunk:
                    if pid not in labels:
                        labels[pid] = pid

        return labels

    def _fetch_entity_labels(self, entity_ids: List[str]) -> Dict[str, str]:
        """
        批量获取实体标签

        参数:
            entity_ids: 实体 ID 列表

        返回:
            dict: {entity_id: label} 映射
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
                logger.warning(f"获取实体标签失败: {e}")
                for eid in chunk:
                    if eid not in labels:
                        labels[eid] = eid

        return labels

    def process_batch_filters(self, changes: List[Dict]):
        """
        处理一批更改记录

        参数:
            changes: 更改记录列表

        返回:
            list: 有效的三元组变化行列表
        """
        # 1. 提取唯一的 QID
        unique_qids = list({c["title"] for c in changes if c["title"].startswith("Q")})
        if not unique_qids:
            return []

        # 2. 批量获取实体详情
        entity_map = {}
        chunks = [unique_qids[i:i + BATCH_SIZE] for i in range(0, len(unique_qids), BATCH_SIZE)]

        def fetch_chunk(chunk_ids):
            """获取一批实体数据"""
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

        # 3. 过滤候选
        candidates = []
        for c in changes:
            qid = c["title"]

            if qid not in entity_map or "missing" in entity_map[qid]:
                continue

            ent = entity_map[qid]

            # 必须有英文维基百科链接
            if "enwiki" not in ent.get("sitelinks", {}):
                continue

            # 过滤掉维基百科分类页面
            enwiki_title = ent.get("sitelinks", {}).get("enwiki", {}).get("title", "")
            if enwiki_title.startswith("Category:"):
                continue

            # 从评论中提取属性 ID
            match = re.search(r"\[\[Property:(P\d+)\]\]", c.get("comment", ""))
            if not match:
                continue

            pid = match.group(1)
            claims = ent.get("claims", {})

            # 确定属性类型
            p_type = "unknown"
            if pid in claims and claims[pid]:
                p_type = claims[pid][0].get("mainsnak", {}).get("datatype", "unknown")

            # 属性类型必须在允许列表中
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

        # 4. 批量获取属性标签
        unique_pids = list({c["pid"] for c in candidates})
        property_labels = self._fetch_property_labels(unique_pids)

        for cand in candidates:
            cand["p_label"] = property_labels.get(cand["pid"], cand["pid"])

        # 5. 批量获取修订版本
        all_revids = set()
        for cand in candidates:
            c = cand["change"]
            if c.get("revid"):
                all_revids.add(str(c["revid"]))
            if c.get("old_revid") and int(c.get("old_revid", 0)) > 0:
                all_revids.add(str(c["old_revid"]))

        revision_cache = self._batch_fetch_revisions(list(all_revids))

        # 6. 收集需要获取标签的对象 ID
        object_ids_to_fetch = set()
        for cand in candidates:
            c = cand["change"]
            qid = c["title"]
            pid = cand["pid"]

            revid_new = str(c.get("revid", ""))
            if revid_new in revision_cache:
                new_val = self._extract_property_value(revision_cache.get(revid_new), pid)
                # 如果是 Wikibase Item（以 Q 开头），添加到待获取列表
                if new_val and isinstance(new_val, str) and new_val.startswith("Q"):
                    object_ids_to_fetch.add(new_val)

        # 批量获取对象标签
        object_labels = self._fetch_entity_labels(list(object_ids_to_fetch))

        # 7. 并行解析差异
        def resolve_diff_cached(item):
            """使用缓存的修订版本解析差异"""
            c = item["change"]
            qid = c["title"]
            pid = item["pid"]

            revid_new = str(c.get("revid", ""))
            revid_old = str(c.get("old_revid", ""))

            try:
                # 获取新旧值
                new_val = self._extract_property_value(revision_cache.get(revid_new), pid)

                if not revid_old or int(c.get("old_revid", 0)) == 0:
                    old_val = "NEW_CREATED"
                else:
                    old_val = self._extract_property_value(revision_cache.get(revid_old), pid)

                # 检查是否有变化
                if new_val != old_val and new_val != "ERROR" and old_val != "ERROR":
                    enwiki_title = item["ent"]["sitelinks"]["enwiki"]["title"]
                    wiki_url = f"https://en.wikipedia.org/wiki/{enwiki_title.replace(' ', '_')}"
                    entity_label = item["ent"].get("labels", {}).get("en", {}).get("value", qid)

                    # 判断变化类型
                    if old_val in ("NEW_CREATED", "NO_VALUE", "SOME_VALUE"):
                        change_type = "created"
                    elif new_val in ("NO_VALUE", "SOME_VALUE"):
                        change_type = "deleted"
                    else:
                        change_type = "updated"

                    # 获取 new_value 的标签
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

        # 并行处理
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
        批量获取修订版本内容

        参数:
            revids: 修订版本 ID 列表

        返回:
            dict: {revid: entity_data} 映射
        """
        if not revids:
            return {}

        revision_cache = {}
        chunks = [revids[i:i + 50] for i in range(0, len(revids), 50)]

        def fetch_chunk(chunk_ids):
            """获取一批修订版本"""
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
                logger.warning(f"获取修订版本失败: {e}")
                return {}

        # 并行获取
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_chunk, chunk): chunk for chunk in chunks}
            for future in concurrent.futures.as_completed(futures):
                revision_cache.update(future.result())

        return revision_cache

    def _extract_property_value(self, entity_data: dict, property_id: str) -> str:
        """
        从实体数据中提取属性值

        参数:
            entity_data: 实体 JSON 数据
            property_id: 属性 ID

        返回:
            str: 属性值（或特殊标记）
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

                # 处理不同数据类型
                if isinstance(val, dict):
                    if "id" in val:
                        return val["id"]  # Wikibase Item
                    if "amount" in val:
                        return val["amount"]  # Quantity
                    if "time" in val:
                        return val["time"]  # Time
                    if "latitude" in val:
                        return f"{val['latitude']},{val['longitude']}"  # Coordinate
                    if "text" in val:
                        return val["text"]  # Monolingual Text

                return str(val)

            elif snaktype == "somevalue":
                return "SOME_VALUE"
            elif snaktype == "novalue":
                return "NO_VALUE"

        except:
            pass

        return "ERROR"

    def run(self):
        """运行提取流程"""
        if self.start_time is not None:
            logger.info(f"获取变更：从 {self.start_time} 到 {self.end_time} (UTC)")
            logger.info(f"时间窗口：{(self.end_time - self.start_time).total_seconds() / 3600:.2f} 小时")
        else:
            logger.info(f"获取变更：从最早记录到 {self.end_time} (UTC)")
            logger.info(f"时间窗口：不限制")
        if self.max_triples:
            logger.info(f"目标三元组数量: {self.max_triples}")

        # 使用字典来去重：key=entity_id, value=最新的变更记录（同一实体只保留一个三元组）
        unique_entities = {}
        total_before_dedup = 0
        start_t = time.time()

        # 逐批处理
        for batch in self.fetch_recent_changes_generator():
            self.total_changes += len(batch)

            valid_rows = self.process_batch_filters(batch)
            if valid_rows:
                total_before_dedup += len(valid_rows)
                for row in valid_rows:
                    entity_id = row[0]  # entity_id
                    timestamp = row[9]  # change_timestamp (现在在索引 9)

                    # 如果是新实体，或者时间戳更新，则更新记录（同一实体只保留最新的那个三元组）
                    if entity_id not in unique_entities or timestamp > unique_entities[entity_id][9]:
                        unique_entities[entity_id] = row

                print(f"已扫描 {self.total_changes} 条变更 | 发现 {len(unique_entities)} 个唯一实体...", end="\r")

                # 如果指定了数量限制且已达到目标，提前退出
                if self.max_triples and len(unique_entities) >= self.max_triples:
                    logger.info(f"\n已达到目标数量 {self.max_triples}，停止扫描")
                    break

        # 去重统计
        logger.info(f"\n去重前: {total_before_dedup} 条变更")
        logger.info(f"去重后: {len(unique_entities)} 个唯一实体")

        # 按时间戳排序（时间戳现在在索引 9）
        sorted_rows = sorted(unique_entities.values(), key=lambda x: x[9], reverse=True)

        # 如果指定了数量限制，只取前 N 个
        if self.max_triples:
            sorted_rows = sorted_rows[:self.max_triples]
            logger.info(f"限制输出: {len(sorted_rows)} 个三元组")

        # 写入文件
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "entity_id", "entity_label", "property_id", "property_label", "property_type",
                "old_value", "new_value", "new_value_label", "change_type", "change_timestamp", "wiki_url"
            ])
            writer.writerows(sorted_rows)

        self.kept_triples = len(sorted_rows)
        duration = time.time() - start_t
        logger.info(f"完成！共扫描 {self.total_changes} 条变更，耗时 {duration:.1f} 秒")
        logger.info(f"保存了 {self.kept_triples} 个有效三元组到 {self.output_file}")


# ========== 主入口 ==========

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从 Wikidata 提取知识三元组变化")
    parser.add_argument("--hours", type=float, default=None,
                        help="扫描时间窗口（小时，默认: 如果指定 --max-triples 则不限制，否则为 2.0）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件路径（默认: outputs/extracted_triples/triple_changes_时间戳.csv）")
    parser.add_argument("--max-triples", type=int, default=None,
                        help="最大输出三元组数量（默认: 不限制，按时间窗口获取所有）")
    args = parser.parse_args()

    # 自动设置时间窗口
    if args.hours is None:
        if args.max_triples:
            # 如果只指定了数量，不限制时间窗口，持续扫描直到找到足够的三元组
            hours = None
        else:
            # 如果都没指定，使用默认的 2 小时
            hours = 2.0
    else:
        hours = args.hours

    # 默认输出路径
    if args.output is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_file = str(EXTRACTED_TRIPLES_DIR / f"triple_changes_{timestamp}.csv")
    else:
        output_file = args.output

    # 打印启动信息
    print("\n" + "=" * 60)
    print("Wikidata 知识三元组变化提取器")
    print("=" * 60)
    if hours is not None:
        print(f"时间窗口: {hours} 小时 (UTC)")
    else:
        print(f"时间窗口: 不限制（持续扫描直到达到目标数量）")
    if args.max_triples:
        print(f"目标数量: {args.max_triples} 个三元组")
    print(f"输出文件: {output_file}")
    print("=" * 60 + "\n")

    # 运行
    extractor = TripleChangeExtractor(hours=hours, output_file=output_file, max_triples=args.max_triples)
    extractor.run()
