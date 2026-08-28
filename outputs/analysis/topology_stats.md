# LiveSearchBench Topology Statistics

## Released Supplement Diversity

| Metric | Value |
|---|---:|
| QA pairs in data description | 1000 |
| Level distribution | 500 L1 / 300 L2 / 200 L3 |
| Full-dataset diversity counts | omitted from metadata |

Note: Subject/relation/object diversity counts are intentionally omitted from this metadata file; compute them from the final release files when preparing full-dataset diversity statistics.

## SPARQL Topology From Input JSON

| Level | Family | n | Share | Avg. edges/constraints | Max edges | Avg. anchors | Unique signatures |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | single-hop | 10 | 33.3% | 1 | 1 | 1 | 1 |
| 2 | multi-constraint intersection | 10 | 33.3% | 2.9 | 5 | 2.9 | 6 |
| 3 | fuzzed/indirect constrained selection | 10 | 33.3% | 2.5 | 4 | 2.5 | 5 |

## Topology Signatures

| Signature | Count |
|---|---:|
| `L1|edges=1|unique_pred=1|anchors=1|repeated_pred=0` | 10 |
| `L2|edges=2|unique_pred=2|anchors=2|repeated_pred=0` | 5 |
| `L3|edges=2|unique_pred=1|anchors=2|repeated_pred=1` | 4 |
| `L3|edges=2|unique_pred=2|anchors=2|repeated_pred=0` | 3 |
| `L2|edges=5|unique_pred=3|anchors=5|repeated_pred=2` | 1 |
| `L2|edges=4|unique_pred=3|anchors=4|repeated_pred=1` | 1 |
| `L3|edges=4|unique_pred=1|anchors=4|repeated_pred=3` | 1 |
| `L3|edges=4|unique_pred=3|anchors=4|repeated_pred=1` | 1 |
| `L2|edges=4|unique_pred=1|anchors=4|repeated_pred=3` | 1 |
| `L2|edges=3|unique_pred=3|anchors=3|repeated_pred=0` | 1 |
| `L2|edges=3|unique_pred=1|anchors=3|repeated_pred=2` | 1 |
| `L3|edges=3|unique_pred=2|anchors=3|repeated_pred=1` | 1 |
