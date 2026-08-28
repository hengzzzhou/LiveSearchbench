# Structured SPARQL Upper-Bound Demo

This diagnostic executes the released canonical SPARQL programs in
`demo.json`, converts count queries into select queries where possible, and
matches returned Wikidata labels against the gold answer string.

| Split | n | Correct | Accuracy |
|---|---:|---:|---:|
| All demo questions | 30 | 27 | 90.0% |
| L1 | 10 | 10 | 100.0% |
| L2 | 10 | 8 | 80.0% |
| L3 | 10 | 9 | 90.0% |

The remaining three misses are cases where the live Wikidata endpoint currently
returns no matching entity or a different entity than the demo gold answer.
This is expected when replaying snapshot-grounded examples against the current
live endpoint, and motivates releasing snapshot identifiers, hashes, cached
SPARQL outputs, and/or a frozen Wikidata-as-corpus diagnostic.
