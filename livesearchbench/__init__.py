"""LiveSearchBench: shared library for benchmark construction and evaluation.

The modules here hold the logic that used to be duplicated across the
``scripts/`` entry points: credential resolution, polite Wikidata access,
dataset loading, and metric computation.

Nothing in this package performs network I/O at import time.
"""

__version__ = "1.1.0"

__all__ = [
    "config",
    "dataio",
    "filters",
    "http",
    "scoring",
    "sparql",
]
