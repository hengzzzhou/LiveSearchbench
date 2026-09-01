#!/usr/bin/env python3
"""Wikidata-as-corpus retrieval: reasoning and retrieval measured separately.

The gold triples behind every instance of a split are verbalised into a passage
corpus, and each question retrieves its own top-k passages from that corpus
before the model ever sees it. Unlike ``oracle.py`` -- which hands an instance
exactly its own evidence -- the model here must find the right passage among
everything the split knows about, and unlike ``RAG.py`` it searches a closed,
noise-free corpus that provably contains the answer.

That makes this the other half of the paper's dual failure mode. Two numbers
are reported side by side:

``retrieval hit rate``
    share of questions whose retrieved context contains the gold answer string.
    This is what the retriever could deliver.

``accuracy``
    share of questions the model then answered correctly.

A low hit rate is a retrieval failure. A high hit rate with low accuracy is a
reasoning failure. Reporting only accuracy cannot tell them apart.

Retrieval
---------
BM25 (Okapi, ``k1=1.5``, ``b=0.75``) is implemented here in pure Python -- no
new dependency, and ``--self-test`` checks it against a hand-computed example.
Retrievers sit behind the small :class:`Retriever` interface, so a dense
retriever can be dropped in by implementing ``index`` and ``retrieve``. **A
dense retriever (Contriever or any other) is deliberately NOT included**: it
would pull in torch and a model download, which this repository does not
require. The BM25 number is therefore a lexical-retrieval baseline, not the
best achievable retrieval.

Caveats worth stating in any write-up: the corpus is built from the split's
own gold triples, so it is tiny (hundreds of passages), free of web noise, and
guaranteed to contain every answer. The hit rate here is an upper bound on
what a real search engine would deliver.

Examples:
    # verify the BM25 implementation against the hand-checked example
    python scripts/eval/wiki_corpus.py --self-test

    # offline: build the corpus, retrieve, report hit rate, never call a model
    python scripts/eval/wiki_corpus.py bench/2025/level1.json \
        --dry-run --top-k 5 --show-retrieval 3

    # real run
    python scripts/eval/wiki_corpus.py bench/2025/level2.json \
        --model gpt-4o-mini --top-k 5 --concurrency 8
"""

from __future__ import annotations

import abc
import argparse
import heapq
import importlib.util
import logging
import math
import re
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from livesearchbench import config, dataio, scoring  # noqa: E402

# The per-level gold-triple extraction lives in oracle.py. It is loaded by path
# under a namespaced module name so that ``scripts/eval`` need not be a package
# and so a same-named third-party module cannot be picked up instead.
_ORACLE_PATH = Path(__file__).resolve().parent / "oracle.py"
_spec = importlib.util.spec_from_file_location("livesearchbench_eval_oracle", _ORACLE_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - only on a broken checkout
    raise ImportError(f"Cannot load the gold-triple extractor from {_ORACLE_PATH}")
oracle = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = oracle
_spec.loader.exec_module(oracle)

logger = logging.getLogger("livesearchbench.eval.wiki_corpus")

METHOD = "wiki_corpus"
DEFAULT_TOP_K = 5
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
DEFAULT_CONCURRENCY = 4
BM25_K1 = 1.5
BM25_B = 0.75

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """Lower-case word tokenisation.

    No stemming and no stop-word list: BM25's IDF term already discounts
    ubiquitous words, and stemming would make the hand-checked ranking in
    :func:`self_test` depend on a stemmer's idiosyncrasies.
    """
    return _TOKEN_RE.findall(str(text or "").lower())


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

class Passage(NamedTuple):
    """One retrievable unit of the verbalised knowledge graph."""

    pid: str
    title: str
    text: str
    item_indices: Tuple[int, ...]


def build_corpus(
    triples_per_item: Sequence[Sequence["oracle.Triple"]],
    *,
    granularity: str = "entity",
) -> Tuple[List[Passage], Dict[int, List[int]]]:
    """Verbalise every gold triple in the split into passages.

    ``granularity="entity"`` groups all triples that share a subject into one
    passage, which is closer to a real document and lets a single passage
    satisfy several constraints of a level-2/3 question. ``"triple"`` emits one
    passage per triple, which is the harder, more fragmented setting.

    Returns the passage list and a map from instance index to the indices of
    the passages that carry that instance's own gold triples.
    """
    if granularity not in ("entity", "triple"):
        raise ValueError(f"unknown granularity {granularity!r}; use 'entity' or 'triple'")

    order: List[str] = []
    lines: Dict[str, List[str]] = {}
    titles: Dict[str, str] = {}
    owners: Dict[str, List[int]] = defaultdict(list)

    for item_index, triples in enumerate(triples_per_item):
        for triple in triples:
            line = oracle.verbalize_triple(triple)
            if granularity == "entity":
                key = triple.subject.lower()
                title = triple.subject
            else:
                key = line.lower()
                title = triple.subject
            if key not in lines:
                order.append(key)
                lines[key] = []
                titles[key] = title
            if line not in lines[key]:
                lines[key].append(line)
            if item_index not in owners[key]:
                owners[key].append(item_index)

    passages: List[Passage] = []
    key_to_index: Dict[str, int] = {}
    for key in order:
        key_to_index[key] = len(passages)
        body = "\n".join(lines[key])
        passages.append(Passage(
            pid=f"p{len(passages):05d}",
            title=titles[key],
            text=f"{titles[key]}\n{body}",
            item_indices=tuple(owners[key]),
        ))

    gold_passages: Dict[int, List[int]] = defaultdict(list)
    for key, index in key_to_index.items():
        for item_index in owners[key]:
            gold_passages[item_index].append(index)
    return passages, dict(gold_passages)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class Retriever(abc.ABC):
    """Minimal retriever interface.

    Implement these two methods to swap BM25 for something else (a dense
    bi-encoder such as Contriever, a hybrid, a reranker). Nothing else in this
    script touches retrieval internals.
    """

    name = "retriever"

    @abc.abstractmethod
    def index(self, passages: Sequence[Passage]) -> None:
        """Build whatever the retriever needs from the corpus."""

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Return ``(passage_index, score)`` pairs, best first.

        ``top_k <= 0`` means "rank the whole corpus". Documents that share no
        term with the query may be omitted, so fewer than ``top_k`` pairs can
        come back. Implementations must be safe to call concurrently from
        several threads once indexed.
        """


class BM25Retriever(Retriever):
    """Okapi BM25 over the verbalised corpus, in pure Python.

    Scoring, for query term ``t`` and document ``d``::

        idf(t)   = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
        score    = sum_t idf(t) * tf(t,d) * (k1 + 1)
                   / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))

    An inverted index keeps scoring proportional to the postings of the query
    terms rather than to the corpus size.
    """

    name = "bm25"

    def __init__(self, *, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = 0
        self.avgdl = 0.0
        self.doc_len: List[int] = []
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}

    def index(self, passages: Sequence[Passage]) -> None:
        self.index_texts([p.text for p in passages])

    def index_texts(self, texts: Sequence[str]) -> None:
        """Index raw strings; used by the corpus path and by the self-test."""
        self.n_docs = len(texts)
        if self.n_docs == 0:
            raise ValueError("cannot index an empty corpus")
        postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_len = []
        for doc_index, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_len.append(len(tokens))
            counts: Dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for token, tf in counts.items():
                postings[token].append((doc_index, tf))
        self.postings = dict(postings)
        self.avgdl = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0
        self.idf = {
            token: math.log(1.0 + (self.n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for token, plist in self.postings.items()
        }

    def retrieve(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.n_docs == 0:
            raise RuntimeError("BM25Retriever.retrieve called before index()")
        scores: Dict[int, float] = defaultdict(float)
        for token in tokenize(query):
            plist = self.postings.get(token)
            if not plist:
                continue
            idf = self.idf[token]
            for doc_index, tf in plist:
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_index] / self.avgdl)
                scores[doc_index] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        limit = self.n_docs if top_k <= 0 else min(top_k, self.n_docs)
        # Sorting on (-score, index) breaks ties by document index, so repeated
        # runs over the same corpus return byte-identical rankings.
        best = heapq.nsmallest(limit, ((-score, index) for index, score in scores.items()))
        return [(index, -negated) for negated, index in best]


def self_test() -> int:
    """Check BM25 against a corpus small enough to score by hand.

    Corpus (N=3, avgdl=6.0), query ``"quick fox"``::

        A: "the quick brown fox"                                   (4 tokens)
        B: "the quick brown fox jumps over the lazy dog the quick"  (11 tokens)
        C: "lazy dog sleeps"                                        (3 tokens)

    ``df(quick) = df(fox) = 2``, so both terms carry
    ``idf = ln(1 + (3-2+0.5)/(2+0.5)) = ln(1.6) = 0.470004``.

    A contains each term once in a short document; B contains ``quick`` twice
    but is nearly three times as long, so length normalisation must push it
    below A. C shares no query term at all, so it scores zero and is not
    returned. The expected result is therefore **A > B**, with
    A = 1.105891, B = 0.871402, and C absent.
    """
    texts = [
        "the quick brown fox",
        "the quick brown fox jumps over the lazy dog the quick",
        "lazy dog sleeps",
    ]
    names = ["A", "B", "C"]
    expected = {"A": 1.1058908923429074, "B": 0.8714023753493280, "C": 0.0}

    retriever = BM25Retriever()
    retriever.index_texts(texts)
    ranking = retriever.retrieve("quick fox", 0)

    print("BM25 self-test: corpus of 3 documents, query 'quick fox'")
    print(f"  N={retriever.n_docs}  avgdl={retriever.avgdl:.4f}  k1={retriever.k1}  b={retriever.b}")
    print(f"  idf(quick)={retriever.idf['quick']:.6f}  idf(fox)={retriever.idf['fox']:.6f}")
    got = {names[i]: score for i, score in ranking}
    for name in names:
        state = "returned" if name in got else "not returned (zero score)"
        print(f"  {name}: score={got.get(name, 0.0):.6f}  expected={expected[name]:.6f}  "
              f"len={len(tokenize(texts[names.index(name)]))}  {state}")
    order = [names[i] for i, _ in ranking]
    print(f"  ranking: {' > '.join(order)}  expected: A > B (C scores zero and is not returned)")

    failures: List[str] = []
    if order != ["A", "B"]:
        failures.append(f"ranking is {order}, expected ['A', 'B']")
    for name in names:
        if abs(got.get(name, 0.0) - expected[name]) > 1e-9:
            failures.append(f"{name} scored {got.get(name, 0.0)!r}, expected {expected[name]!r}")
    if failures:
        for message in failures:
            print(f"  FAIL: {message}")
        return 1
    print("  PASS: ranking and all three scores match the hand computation.")
    return 0


# ---------------------------------------------------------------------------
# Prompting and evaluation
# ---------------------------------------------------------------------------

def format_passages(passages: Sequence[Passage], hits: Sequence[Tuple[int, float]]) -> str:
    blocks = []
    for rank, (index, score) in enumerate(hits, 1):
        passage = passages[index]
        blocks.append(f"[{rank}] (score {score:.3f}) {passage.text}")
    return "\n\n".join(blocks) if blocks else "(no passages were retrieved)"


def build_prompt(question: str, context: str) -> str:
    if not question.endswith("?"):
        question += "?"
    return (
        "You are given passages retrieved from a knowledge base of Wikidata facts. Each passage "
        "lists facts about one entity in the form subject | property | value. Some passages may be "
        "irrelevant.\n\n"
        "Retrieved passages:\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the retrieved passages. Think step by step, then provide ONLY "
        "the final answer inside <answer> and </answer> tags. Do not output anything else after "
        "the tags. For example, <answer> Beijing </answer>."
    )


def evaluate_item(
    item: Dict,
    *,
    passages: Sequence[Passage],
    hits: Sequence[Tuple[int, float]],
    gold_rank: Optional[int],
    client,
    model_name: str,
    max_tokens: int,
    temperature: float,
    dry_run: bool,
) -> Dict:
    question = oracle.clean_text(item.get("question"))
    expected = item.get("answer", "")
    aliases = item.get("answer_aliases") or []
    context = format_passages(passages, hits)
    prompt = build_prompt(question, context)

    record: Dict = {
        "question": question,
        "expected_answer": expected,
        # Carried through so a later re-score by scripts/analysis reproduces
        # exactly the numbers written here.
        "answer_aliases": list(aliases),
        "level": item.get("level"),
        "retrieved_passage_ids": [passages[i].pid for i, _ in hits],
        "retrieved_titles": [passages[i].title for i, _ in hits],
        "retrieval_scores": [round(s, 4) for _, s in hits],
        # The metric that separates a retrieval failure from a reasoning one.
        "retrieval_hit": scoring.contains_match(context, expected, aliases),
        "gold_passage_rank": gold_rank,
        "retrieved_context": context,
    }

    if dry_run:
        record.update({
            "model_answer": "", "reasoning_process": "", "is_correct": False,
            "dry_run": True, "error": None,
        })
        record.update({k: 0.0 for k in ("exact_match", "token_f1", "contains_match")})
        return record

    try:
        content = oracle.call_model(
            client, [{"role": "user", "content": prompt}], model_name=model_name,
            max_tokens=max_tokens, temperature=temperature,
        )
        error = None
    except RuntimeError as exc:
        logger.error("Giving up on question %r: %s", question[:60], exc)
        content = ""
        error = str(exc)

    answer = oracle.extract_answer(content)
    scores = scoring.score_item(answer, expected, aliases)
    record.update({
        "model_answer": answer,
        "reasoning_process": content,
        # Normalised exact match, matching DA/CoT/RAG. Previously this was
        # contains_match, so accuracy from this runner was not comparable
        # with theirs. contains_match is still recorded alongside.
        "is_correct": bool(scores["exact_match"]),
        "dry_run": False,
        "error": error,
    })
    record.update(scores)
    return record


def build_summary(results: Sequence[Dict], *, model_name: str, dry_run: bool, top_k: int,
                  corpus_size: int, retriever_name: str, granularity: str) -> Dict:
    total = len(results)
    scored = [r for r in results if not r.get("dry_run")]
    correct = sum(1 for r in scored if r.get("is_correct"))
    hits = sum(1 for r in results if r.get("retrieval_hit"))
    gold_ranks = [r["gold_passage_rank"] for r in results if r.get("gold_passage_rank")]
    gold_in_top_k = sum(1 for r in gold_ranks if r <= top_k)
    hit_rate = (hits / total) if total else 0.0
    accuracy = (correct / len(scored)) if scored else None
    return {
        "method": METHOD,
        "model": model_name,
        "dry_run": dry_run,
        "retriever": retriever_name,
        "top_k": top_k,
        "passage_granularity": granularity,
        "corpus_passages": corpus_size,
        "total_questions": total,
        "scored_questions": len(scored),
        "correct_answers": correct,
        "accuracy": accuracy,
        "errors": sum(1 for r in results if r.get("error")),
        "retrieval_hit_rate": hit_rate,
        "gold_passage_recall_at_k": (gold_in_top_k / total) if total else 0.0,
        "gold_passage_found": len(gold_ranks),
        "mean_gold_passage_rank": (sum(gold_ranks) / len(gold_ranks)) if gold_ranks else None,
        # Of the questions whose context did contain the answer, how many were
        # answered correctly. This is the reasoning half of the failure mode.
        "accuracy_given_hit": (
            (sum(1 for r in scored if r.get("retrieval_hit") and r.get("is_correct"))
             / max(1, sum(1 for r in scored if r.get("retrieval_hit"))))
            if scored and any(r.get("retrieval_hit") for r in scored) else None
        ),
        "metrics": scoring.aggregate(scored) if scored else {},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wiki_corpus.py",
        description="Wikidata-as-corpus retrieval: verbalise a split's gold triples into a "
                    "passage corpus, retrieve top-k per question with BM25, and report retrieval "
                    "hit rate alongside answer accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A low hit rate means retrieval failed; a high hit rate with low accuracy means "
               "reasoning failed. Run --self-test to verify the BM25 implementation.",
    )
    parser.add_argument("data_positional", nargs="?", metavar="DATA",
                        help="benchmark split, e.g. bench/2025/level1.json or demo.json")
    parser.add_argument("--data", dest="data_flag", metavar="PATH",
                        help="same as the positional DATA argument")
    parser.add_argument("--model", default="", help="model name passed to the chat completions API")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"passages placed in the prompt (default: {DEFAULT_TOP_K})")
    parser.add_argument("--retriever", choices=["bm25"], default="bm25",
                        help="retrieval backend; dense retrievers (e.g. Contriever) are not "
                             "shipped, implement the Retriever interface to add one")
    parser.add_argument("--granularity", choices=["entity", "triple"], default="entity",
                        help="one passage per subject entity (default) or per single triple")
    parser.add_argument("--concurrency", "--threads", dest="concurrency", type=int,
                        default=DEFAULT_CONCURRENCY, help="parallel model calls (default: 4)")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help="sampling temperature (default: 0.7)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="maximum tokens to generate per answer (default: 2048)")
    parser.add_argument("--limit", type=int, default=0,
                        help="evaluate only the first N instances (0 = all)")
    parser.add_argument("--resolve-labels", action="store_true",
                        help="look up English labels for Q/P ids recovered from SPARQL programs; "
                             "required for demo.json, otherwise its passages are opaque ids")
    parser.add_argument("--label-cache", default=str(oracle.DEFAULT_LABEL_CACHE),
                        help="label cache file shared with oracle.py")
    parser.add_argument("--output-dir", default=None,
                        help="where to write results (default: outputs/evaluations/<year>)")
    parser.add_argument("--resume", nargs="?", const="", default=None, metavar="RESULTS_JSON",
                        help="skip questions already answered. With no value, reuses the newest "
                             "matching results file in the output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="build the corpus and retrieve for real, but never call the model; "
                             "needs no API key and still reports the retrieval hit rate")
    parser.add_argument("--show-retrieval", type=int, default=0, metavar="N",
                        help="print the top-3 retrieved passages for the first N questions")
    parser.add_argument("--save-corpus", default=None, metavar="PATH",
                        help="also write the verbalised passage corpus to this JSON file")
    parser.add_argument("--self-test", action="store_true",
                        help="verify BM25 against a hand-computed example and exit")
    parser.add_argument("--base-url", default=None, help="override OPENAI_BASE_URL")
    parser.add_argument("--api-key", default=None, help="override OPENAI_API_KEY")
    parser.add_argument("--verbose", "-v", action="store_true", help="debug logging")
    args = parser.parse_args(argv)

    if args.self_test:
        return args
    args.data = args.data_flag or args.data_positional
    if not args.data:
        parser.error("a dataset path is required (positional DATA or --data), or use --self-test")
    if not args.dry_run and not args.model:
        parser.error("--model is required unless --dry-run is given")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.self_test:
        return self_test()

    raw_items, meta = dataio.load_instances(args.data)
    items = [dataio.normalize_instance(it) for it in raw_items]
    if args.limit:
        items = items[:args.limit]
    logger.info("Loaded %d instance(s) from %s (level=%s year=%s)",
                len(items), args.data, meta.get("level"), meta.get("year"))

    labels: Dict[str, str] = {}
    if args.resolve_labels:
        needed: List[str] = []
        for item in items:
            provisional = oracle.extract_gold_triples(item)
            if any(t.source == "sparql_verification" for t in provisional):
                needed.extend(oracle.wikidata_ids(item.get("sparql_verification", "")))
        labels = oracle.resolve_labels(needed, cache_path=args.label_cache) if needed else {}
    else:
        labels = oracle.load_label_cache(args.label_cache)

    triples_per_item = [oracle.extract_gold_triples(item, labels=labels) for item in items]
    oracle.warn_if_unresolved(triples_per_item, resolved=args.resolve_labels)
    empty = sum(1 for t in triples_per_item if not t)
    if empty:
        logger.warning("%d/%d instance(s) contributed no triples to the corpus", empty, len(items))

    passages, gold_passages = build_corpus(triples_per_item, granularity=args.granularity)
    logger.info("Corpus: %d passage(s) from %d instance(s) at %s granularity",
                len(passages), len(items), args.granularity)
    if args.save_corpus:
        import json

        path = dataio.ensure_parent(args.save_corpus)
        path.write_text(json.dumps([p._asdict() for p in passages], ensure_ascii=False, indent=2),
                        encoding="utf-8")
        logger.info("Wrote corpus to %s", path)

    retriever: Retriever = BM25Retriever()
    retriever.index(passages)

    # The split corpora hold a few hundred passages, so the full ranking is
    # cheap and gives the gold passage's rank for free alongside the top-k.
    rankings: List[List[Tuple[int, float]]] = []
    gold_ranks: List[Optional[int]] = []
    for index, item in enumerate(items):
        full = retriever.retrieve(item["question"], 0)
        rankings.append(full[:args.top_k])
        wanted = set(gold_passages.get(index, ()))
        rank = next((r for r, (doc, _) in enumerate(full, 1) if doc in wanted), None)
        gold_ranks.append(rank)

    for index in range(min(args.show_retrieval, len(items))):
        print(f"\n===== retrieval {index + 1}/{len(items)} =====")
        print(f"Question   : {items[index]['question']}")
        print(f"Gold answer: {items[index]['answer']}")
        print(f"Gold passage rank: {gold_ranks[index]}")
        for rank, (doc, score) in enumerate(rankings[index][:3], 1):
            body = passages[doc].text.replace("\n", " / ")
            print(f"  [{rank}] score={score:.3f} pid={passages[doc].pid} {body}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(
        REPO_ROOT / "outputs" / "evaluations" / str(meta.get("year") or "unknown"))
    output_dir.mkdir(parents=True, exist_ok=True)

    done: Dict[str, Dict] = {}
    if args.resume is not None:
        resume_path = Path(args.resume) if args.resume else oracle.find_resume_file(
            output_dir, args.model, str(meta.get("level") or "unknown"), method=METHOD)
        if resume_path and Path(resume_path).is_file():
            done = oracle.load_resume(Path(resume_path))
        else:
            logger.warning("--resume given but no prior results file was found in %s", output_dir)

    client = None
    if not args.dry_run:
        base_url, api_key = config.openai_credentials(base_url=args.base_url, api_key=args.api_key)
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info("Calling model %r at %s with concurrency %d",
                    args.model, base_url, args.concurrency)
    else:
        logger.warning("DRY RUN: retrieval runs for real, but no model is called; accuracy is "
                       "reported as null while the retrieval hit rate is a real measurement")

    counter = {"n": 0}
    counter_lock = threading.Lock()

    def run_one(payload: Tuple[int, Dict]) -> Dict:
        index, item = payload
        cached = done.get(oracle.clean_text(item.get("question")))
        if cached is not None:
            return cached
        record = evaluate_item(
            item, passages=passages, hits=rankings[index], gold_rank=gold_ranks[index],
            client=client, model_name=args.model, max_tokens=args.max_tokens,
            temperature=args.temperature, dry_run=args.dry_run,
        )
        with counter_lock:
            counter["n"] += 1
            logger.info("[%d/%d] hit=%s %s -> %r (gold %r)", counter["n"], len(items),
                        record["retrieval_hit"],
                        "dry-run" if args.dry_run else ("correct" if record["is_correct"] else "wrong"),
                        record["model_answer"][:60], record["expected_answer"])
        return record

    payloads = list(enumerate(items))
    if args.dry_run or args.concurrency == 1:
        results = [run_one(p) for p in payloads]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results = list(pool.map(run_one, payloads))

    summary = build_summary(results, model_name=args.model or "(dry-run)", dry_run=args.dry_run,
                            top_k=args.top_k, corpus_size=len(passages),
                            retriever_name=retriever.name, granularity=args.granularity)
    # dataio.save_run merges this metadata OVER the summary, and every released
    # split's header already defines total_questions, so the split's own header
    # is nested rather than spread to keep the run's counts intact.
    run_meta = {"level": meta.get("level"), "year": meta.get("year"),
                "dataset_metadata": meta}
    paths = dataio.save_run(
        results=results, summary=summary, method=METHOD,
        model_name=args.model or "dry_run", data_path=args.data,
        output_dir=str(output_dir), metadata=run_meta,
    )

    print("\n=== Wikidata-as-corpus retrieval ===")
    print(f"retriever / top-k        : {summary['retriever']} / {summary['top_k']} "
          f"({summary['passage_granularity']} passages)")
    print(f"corpus passages          : {summary['corpus_passages']}")
    print(f"instances                : {summary['total_questions']}")
    print(f"retrieval hit rate       : {summary['retrieval_hit_rate']:.1%}  "
          f"(gold answer string present in the retrieved context)")
    print(f"gold passage recall@{summary['top_k']:<4} : {summary['gold_passage_recall_at_k']:.1%}")
    if summary["mean_gold_passage_rank"] is not None:
        # The mean is over retrieved gold passages only, so print the
        # denominator whenever some gold passage was never ranked at all.
        print(f"mean gold passage rank   : {summary['mean_gold_passage_rank']:.2f} "
              f"(over the {summary['gold_passage_found']}/{summary['total_questions']} "
              f"question(s) whose gold passage was ranked at all)")
    if summary["accuracy"] is None:
        print("accuracy                 : n/a (dry run, no model was called)")
    else:
        print(f"accuracy (contains)      : {summary['accuracy']:.2%} "
              f"({summary['correct_answers']}/{summary['scored_questions']})")
        if summary["accuracy_given_hit"] is not None:
            print(f"accuracy | retrieval hit : {summary['accuracy_given_hit']:.2%}")
        overall = summary["metrics"].get("overall", {})
        for metric in ("exact_match", "token_f1", "contains_match"):
            cell = overall.get(metric, {})
            print(f"{metric:<25}: {cell.get('value')} "
                  f"[{cell.get('ci_low')}, {cell.get('ci_high')}]")
    print(f"results                  : {paths['results']}")
    print(f"summary                  : {paths['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
