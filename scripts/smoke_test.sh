#!/usr/bin/env bash
# End-to-end offline check of the whole LiveSearchBench pipeline.
#
# Runs every stage against the bundled fixtures with a stub model: no API keys,
# no network, under a minute. If this passes, a fresh clone works.
#
#   ./scripts/smoke_test.sh
#
# Each stage prints the command it runs, so this doubles as a worked example.

set -euo pipefail

PYTHON="${PYTHON:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

step=0
run() {
    step=$((step + 1))
    printf '\n\033[1m[%d/%d] %s\033[0m\n' "$step" "$TOTAL" "$1"
    shift
    printf '    $ %s\n' "$*"
    if ! "$@" > "$WORK/out.txt" 2>&1; then
        printf '\n  FAILED. Output:\n'
        sed 's/^/      /' "$WORK/out.txt"
        exit 1
    fi
    grep -viE 'NotOpenSSL|warnings\.warn' "$WORK/out.txt" | tail -n "${TAIL:-4}" | sed 's/^/      /'
}

TOTAL=8
printf '\033[1mLiveSearchBench smoke test\033[0m  (offline, no API keys)\n'

run "Unit tests" \
    "$PYTHON" -m pytest tests/ -q

run "Dataset description matches the released files" \
    "$PYTHON" scripts/analysis/dataset_stats.py --check

run "Step 1: dump differential (paper Delta_plus / Delta_circle)" \
    "$PYTHON" scripts/extract_dump_delta.py \
        --t0 data/sample/dump_T0.json.gz --t1 data/sample/dump_T1.json.gz \
        --output "$WORK/delta.csv"

TAIL=3 run "Step 3: Level 1 question synthesis (stub model, real SPARQL skipped)" \
    "$PYTHON" scripts/generate_level1.py --input data/sample/triple_changes_sample.csv \
        --dry-run --skip-verification --num-questions 4 --candidate-pool 20 \
        --seed 0 --year 2025 --output "$WORK/level1.json"

# bench/2021/level2.json carries constraint_info, so the verifier has real
# provenance to work from. demo.json has none and would need --resolve-sparql.
TAIL=6 run "Step 4: contextual solvability verification" \
    "$PYTHON" scripts/verify_solvability.py bench/2021/level2.json --dry-run --limit 10 \
        --verdicts "$WORK/verdicts.json" --filtered "$WORK/passed.json"

TAIL=6 run "Evaluation: closed-book direct answer" \
    "$PYTHON" scripts/eval/DA.py demo.json --dry-run --limit 10 \
        --output-dir "$WORK/evals"

TAIL=5 run "Diagnostic: BM25 Wikidata-as-corpus retrieval" \
    "$PYTHON" scripts/eval/wiki_corpus.py bench/2025/level3.json --dry-run --top-k 5 \
        --output-dir "$WORK/evals"

TAIL=8 run "Scoring: exact match, token F1 and bootstrap CIs" \
    "$PYTHON" scripts/analysis/score.py "$WORK/evals" --format table

printf '\n\033[1;32mAll %d stages passed.\033[0m\n' "$TOTAL"
printf 'The pipeline runs end to end. Add API keys (see .env.example) for real runs.\n'
