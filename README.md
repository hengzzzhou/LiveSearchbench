# LiveSearchbench

Lightweight scripts to extract recent Wikidata triple changes and generate evaluation questions across three difficulty levels, plus runners for QA evaluation.

## Prerequisites
- Python 3.9+ with `requests`, `pandas`, and `openai` (install via `pip install -r requirements.txt` if present, or install packages manually).
- API keys must be placed in the script globals before running generation/eval scripts:
  - `scripts/generate_level*.py`: set `API_KEY`, `API_BASE_URL`, `API_MODEL`.
  - `scripts/eval/*.py`: set `OPENAI_BASE_URL`, `OPENAI_API_KEY`; for `RAG.py` also set `SERPER_*` fields.

## Scripts & examples
- Extract recent triple changes from Wikidata  
  `python scripts/extract_triple_changes.py --hours 2 --output outputs/extracted_triples/triple_changes.csv`

- Generate Level 1 questions from triples (CSV from extractor)  
  `python scripts/generate_level1.py --input outputs/extracted_triples/triple_changes.csv`

- Generate Level 2 multi-attribute questions  
  `python scripts/generate_level2.py --input outputs/extracted_triples/triple_changes.csv`

- Generate Level 3 advanced questions using triples and Level 2 output  
  `python scripts/generate_level3.py --input outputs/extracted_triples/triple_changes.csv --level2 outputs/questions/level2_multi_attribute_only.json`

- Evaluate QA with iterative search/RAG (provide dataset JSON)  
  `python scripts/eval/RAG.py data/level3_questions.json --model your-model-name --serper-key $SERPER_KEY`

- Evaluate QA with chain-of-thought only (no search)  
  `python scripts/eval/CoT.py data/level3_questions.json --model your-model-name`

- Evaluate QA with direct-answer only (no search)  
  `python scripts/eval/DA.py data/level3_questions.json --model your-model-name`
