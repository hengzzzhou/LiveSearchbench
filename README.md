# LiveSearchBench

An automated pipeline that builds **continually updating, retrieval-dependent QA benchmarks** from
Wikidata snapshot differentials. Every question is grounded in a fact that changed between two
snapshots, and every answer is validated as unique by a SPARQL program that ships with the instance.

This repository contains both halves of the work: the **released benchmark splits**, and the
**pipeline that regenerates them**, so you can evaluate on our data or build your own from a newer
pair of snapshots.

```bash
git clone https://github.com/hengzzzhou/LiveSearchbench && cd LiveSearchbench
pip install -r requirements-dev.txt
make smoke          # end-to-end check: no API keys, no network, ~4 seconds
```

If `make smoke` passes, everything below works.

---

## The released data

| Split | L1 | L2 | L3 | Total |
|---|---:|---:|---:|---:|
| `bench/2021/` | 150 | 100 | 50 | 300 |
| `bench/2025/` | 150 | 100 | 50 | 300 |
| **All** | **300** | **200** | **100** | **600** |

`demo.json` holds 30 mixed examples for a quick look. `data_description.json` carries the full
statistics and is **generated, never hand-edited** — regenerate it with
`python scripts/analysis/dataset_stats.py --write`, and CI fails if it drifts from the data.

Difficulty tiers: **L1** single-hop; **L2** multi-constraint intersection (several constraints that
jointly single out one entity); **L3** the L2 question rewritten with indirect, abstracted
descriptions.

Instances come in three JSON shapes across the released files (a bare list, a `metadata` wrapper and
a `dataset_info` wrapper). You never need to care: `livesearchbench.dataio.load_instances` reads all
of them, and so does every script here.

```python
from livesearchbench import dataio
instances, meta = dataio.load_instances("bench/2025/level2.json")
```

### Level 3 answer verification, stated precisely

L3 questions **inherit their SPARQL program from their L2 parent** — the abstraction step rewrites
the question text, not the query. Every L3 instance therefore carries
`"sparql_verification_source": "inherited_from_level2"` plus a `verification` block recording the
mode used, the endpoint, the timestamp and the observed result count. `answer_verified` reports what
was actually checked. See `--verify` under [Building your own](#building-your-own-benchmark).

---

## Setup

```bash
pip install -r requirements.txt          # runtime: requests, openai, pandas
pip install -r requirements-dev.txt      # adds pytest
cp .env.example .env                     # then fill in your keys
```

Credentials resolve in the order **CLI flag → environment variable → `.env` → default**, handled by
`livesearchbench.config`. No key is ever stored in a script. A missing key produces a message naming
the variable and every way to set it, rather than an opaque HTTP error later.

Only two keys matter: `OPENAI_API_KEY` (any OpenAI-compatible endpoint via `OPENAI_BASE_URL`) and
`SERPER_API_KEY` for web search in `RAG.py`. **Every script also accepts `--dry-run`**, which swaps
the model call for a deterministic stub so the whole pipeline runs with no keys at all.

---

## Evaluating on LiveSearchBench

Three runners, one record format, one scorer.

```bash
# closed-book
python scripts/eval/DA.py  bench/2025/level1.json --model gpt-4o-mini
python scripts/eval/CoT.py bench/2025/level2.json --model gpt-4o-mini --threads 8

# retrieval-augmented, iterative search via Serper
python scripts/eval/RAG.py bench/2025/level3.json --model gpt-4o-mini --max-iter 5

# no keys needed
python scripts/eval/DA.py demo.json --dry-run
```

Shared flags: `--limit`, `--threads`, `--temperature`, `--n-samples N` (for pass@k), `--resume`
(skips questions already answered, so an interrupted run continues), `--output-dir`, `--dry-run`.
`RAG.py` adds `--max-iter`, `--serper-key`, `--serper-endpoint`.

Runs write incrementally, so killing one does not lose the work already done.

### Scoring

```bash
python scripts/analysis/score.py outputs/evaluations/2025/*_results.json
python scripts/analysis/score.py outputs/evaluations/2025/*_results.json --format markdown
python scripts/analysis/score.py run_results.json --pass-at-k 1 --pass-at-k 4 --pass-at-k 8
```

The scorer reports four columns side by side, overall and per level, each with a **95% bootstrap
confidence interval**:

| column | meaning |
|---|---|
| **EM** | normalised exact match — case-folded, articles and punctuation stripped |
| **F1** | SQuAD-style token F1 |
| **Contains** | case-folded substring containment (`gold in prediction`) |
| **Recorded** | the `is_correct` flag exactly as the runner stored it |

`Contains` is strictly more permissive than `EM`: a rambling answer that merely mentions the gold
string counts as correct under it. Earlier versions of this repository scored with containment under
the name `simple_match`, so both are reported and the gap is visible rather than hidden.
`python scripts/analysis/score.py --self-test` runs 21 checks demonstrating exactly this.

### Diagnostics: is it a retrieval failure or a reasoning failure?

```bash
# Oracle: inject the gold triples, skip retrieval entirely -> measures reasoning alone
python scripts/eval/oracle.py bench/2025/level2.json --model gpt-4o-mini

# Wikidata-as-corpus: BM25 over passages built from the split's own gold triples
python scripts/eval/wiki_corpus.py bench/2025/level3.json --model gpt-4o-mini --top-k 5
python scripts/eval/wiki_corpus.py --self-test        # BM25 vs a hand-computed ranking
```

`wiki_corpus.py` reports **retrieval hit rate** next to accuracy. A low hit rate is a retrieval
failure; a high hit rate with low accuracy is a reasoning failure. Measured with `--dry-run`
(retrieval is real, no model is called):

| split | passages | hit rate @5 | mean gold rank |
|---|---:|---:|---:|
| `bench/2025/level1.json` | 149 | 100.0% | 1.00 |
| `bench/2025/level3.json` | 50 | 66.0% | 5.04 |

BM25 is Okapi (`k1=1.5`, `b=0.75`) in pure Python — no extra dependency — behind a small `Retriever`
interface. **Contriever is not included**; it would require torch and a model download.

Because the corpus is built from the split's own gold triples it is small and noise-free, so these
hit rates are an *upper bound* on what a live search engine would return.

---

## Building your own benchmark

Four stages. Stages 1 and 2 give you a triple CSV; stages 3 and 4 turn it into verified questions.

### Stage 1 — get a knowledge delta

Two extractors, because they answer different questions.

**From two dumps** (this is what the paper describes: Δ⁺ insertions ∪ Δ° updates). Use this for any
historical window.

```bash
python scripts/extract_dump_delta.py \
    --t0 dumps/wikidata-20250501-all.json.gz \
    --t1 dumps/wikidata-20250801-all.json.gz \
    --output outputs/extracted_triples/delta_2025.csv --workers 8
```

It streams both dumps line by line and indexes T0 into sqlite, so memory stays flat regardless of
dump size. It also writes a **MANIFEST** recording each dump's path, size, mtime and SHA-256 — the
provenance you need to make a release reproducible.

> Wikimedia keeps only ~8 weeks of dumps at `dumps.wikimedia.org`. Older ones are on the Internet
> Archive under identifiers like `wikibase-wikidatawiki-20211231`. Record both in your manifest.

**From live recent changes** — convenient, but bounded by Wikimedia's ~30-day `recentchanges`
retention. The script refuses windows beyond that instead of silently truncating.

```bash
python scripts/extract_triple_changes.py --hours 6 --output outputs/extracted_triples/recent.csv
```

Both write the same 11-column CSV, so stage 3 accepts either.

### Stage 2 — filtering

Applied inside both extractors, and reported as a funnel:

1. **Relation allow-list** — 198 curated relation labels, plus a deny-list of 31 meta/formatting
   property IDs (`P18` image, `P31` instance of, `P279` subclass of, …). Disable the allow-list with
   `--no-allowlist`.
2. **Entity quality** — requires an English Wikipedia sitelink, drops disambiguation pages, list
   articles and Wikimedia-namespace surface forms.
3. **Statement validity** — drops `deprecated` rank, prefers `preferred`, deduplicates by statement
   ID (falling back to a normalised `(subject, relation)` key).

Every run writes `<output>_filter_stats.json` / `.md` counting what entered each stage and why
things were dropped, so the pipeline funnel is reproducible rather than asserted.

### Stage 3 — question synthesis

```bash
python scripts/generate_level1.py --input outputs/extracted_triples/delta_2025.csv \
    --model gpt-4o --num-questions 300 --seed 0 --year 2025
python scripts/generate_level2.py --input outputs/extracted_triples/delta_2025.csv \
    --model gpt-4o --num-questions 300 --seed 0 --year 2025
python scripts/generate_level3.py --input outputs/extracted_triples/delta_2025.csv \
    --level2 outputs/questions/level2_300_questions_2025.json \
    --model gpt-4o --num-questions 200 --seed 0 --verify count
```

Every generator takes `--model`, `--seed`, `--output`, `--num-questions`, `--candidate-pool` and
`--dry-run`, and **stamps the model actually used, the seed and the real counts into the output
metadata**, so a released file can never misreport how it was made. Uniqueness is enforced by
SPARQL `COUNT = 1` against a configurable endpoint (`--endpoint`).

`generate_level3.py --verify` controls what `answer_verified` means:

| mode | behaviour |
|---|---|
| `count` (default) | re-runs the inherited COUNT program; verified iff it returns 1 |
| `answer` | as `count`, and the returned entity's English label must match the gold string |
| `inherit` | copies the L2 parent's flag, running no check |
| `skip` | runs no check; `answer_verified` is `false` everywhere |

### Stage 4 — contextual solvability verification

The quality gate: give a verifier model the question **plus its gold triples and nothing else**, with
no retrieval, and keep the instance only if it derives the gold answer. This is what rules out
ambiguous or hallucinated questions.

```bash
python scripts/verify_solvability.py outputs/questions/level2_300_questions_2025.json \
    --model Qwen3-235B-A22B-Instruct-2507 --n-votes 3 --concurrency 8 \
    --verdicts outputs/verification/l2_verdicts.json \
    --filtered outputs/verification/l2_verified.json
```

It writes a per-instance verdict file *and* a filtered split containing only what passed, so it
works as a real pipeline filter. The prompt is a module-level constant (`VERIFIER_PROMPT`) so it can
be quoted and audited. Pass rates are reported per level with bootstrap CIs.

---

## Repository layout

```
livesearchbench/       shared library (see below)
scripts/
  extract_dump_delta.py        stage 1: two dumps -> delta CSV (+ manifest, funnel)
  extract_triple_changes.py    stage 1: live recent changes -> delta CSV
  generate_level{1,2,3}.py     stage 3: delta CSV -> verified questions
  verify_solvability.py        stage 4: gold-context verification filter
  eval/{DA,CoT,RAG}.py         evaluation runners
  eval/oracle.py               oracle triple injection
  eval/wiki_corpus.py          BM25 Wikidata-as-corpus retrieval
  eval/SPARQLUpperBound.py     replays the released SPARQL programs
  analysis/score.py            EM / F1 / containment + bootstrap CIs + pass@k
  analysis/dataset_stats.py    regenerates data_description.json
  analysis/topology_stats.py   query-graph topology of the released programs
  analysis/summarize_rag_budget.py   accuracy vs search budget
  smoke_test.sh                the 8-stage offline end-to-end check
bench/, demo.json      the released benchmark
data/sample/           fixtures the smoke test runs on
tests/                 105 unit tests
```

### The library

`livesearchbench/` holds the logic the scripts share. It is useful on its own:

| module | what it gives you |
|---|---|
| `config` | credential resolution with actionable errors; policy-compliant User-Agent |
| `http` | `PoliteSession`: bounded retries, exponential backoff, `maxlag`, always a timeout |
| `sparql` | `SparqlClient`; `count()` **raises** on failure rather than returning 0 |
| `dataio` | loads every released file shape; `save_run` / `load_results` |
| `scoring` | `exact_match`, `token_f1`, `contains_match`, `bootstrap_ci`, `pass_at_k` |
| `filters` | the relation allow-list, entity checks, rank filtering, funnel counters |

---

## Development

```bash
make test     # 105 unit tests
make smoke    # 8-stage offline end-to-end run
make score    # score whatever runs are in outputs/evaluations/
make help
```

CI runs the tests on Python 3.9/3.11/3.12, checks that `data_description.json` still matches the
data, and runs the smoke test. A separate weekly job exercises the live Wikidata path, so an
endpoint change surfaces here rather than in your clone.

---

## Licence and attribution

| component | licence |
|---|---|
| Code (`scripts/`, `livesearchbench/`, `tests/`) | MIT — see `LICENSE` |
| Benchmark data (`bench/`, `demo.json`, `data/`) | CC BY 4.0 — see `LICENSE-DATA` |
| Upstream facts (Wikidata) | CC0 1.0 |

Data from Wikidata. Wikidata's structured data is CC0 and imposes no attribution requirement; we
credit it because the community asks and because it is good practice.

**The questions in this benchmark were synthesised by large language models** from Wikidata triples,
then validated automatically and reviewed by human annotators. They are not human-written text. Full
details in `NOTICE`.

LiveSearchBench is an **evaluation** resource. Please do not train on it — that defeats the point of
a contamination-resistant benchmark.

## Citation

See `CITATION.cff`. To appear in Findings of the Association for Computational Linguistics: EMNLP 2026.
