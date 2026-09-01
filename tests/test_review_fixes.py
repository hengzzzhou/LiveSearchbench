"""Regression tests for the defects found in the pre-release review.

Each test pins behaviour that was wrong in the first release, so a future change
that reintroduces the bug fails here rather than in somebody's benchmark run.
"""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from livesearchbench import filters, scoring, sparql

ROOT = Path(__file__).resolve().parent.parent


def _load(script):
    spec = importlib.util.spec_from_file_location(Path(script).stem, ROOT / script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestPassAtKDomain:
    """pass_at_k(2, 0, 3) used to return 1.0 for an item never answered right."""

    def test_k_above_n_samples_raises(self):
        with pytest.raises(ValueError, match="undefined"):
            scoring.pass_at_k(2, 0, 3)

    def test_valid_domain_still_works(self):
        assert scoring.pass_at_k(4, 1, 1) == pytest.approx(0.25)
        assert scoring.pass_at_k(4, 1, 4) == 1.0

    def test_rejects_nonsense_inputs(self):
        for args in [(0, 0, 1), (4, 5, 1), (4, 1, 0), (4, -1, 1)]:
            with pytest.raises(ValueError):
                scoring.pass_at_k(*args)


class TestUnicodeNormalisation:
    """Normalisation was ASCII-only: string.punctuation and lower()."""

    @pytest.mark.parametrize("a,b", [
        ("Baden–Württemberg", "Baden-Württemberg"),   # en dash vs hyphen
        ("Straße", "STRASSE"),                        # casefold, not lower
        ("北京。", "北京"),                             # CJK full stop
        ("«Paris»", "Paris"),                         # guillemets
    ])
    def test_equivalent_surface_forms_match(self, a, b):
        assert scoring.exact_match(a, b)

    def test_diacritics_are_still_significant(self):
        # SQuAD-style normalisation does not strip accents, and loosening this
        # would change every number already reported.
        assert not scoring.exact_match("café", "cafe")


class TestBootstrapSymmetry:
    def test_interval_is_symmetric_in_rank(self):
        vals = [100.0] * 30 + [0.0] * 70
        ci = scoring.bootstrap_ci(vals, resamples=10000, seed=0)
        assert ci["lo"] < ci["mean"] < ci["hi"]
        # Binomial 95% CI for 30/100 is about [21, 40].
        assert 18 <= ci["lo"] <= 24 and 36 <= ci["hi"] <= 42


class TestSparqlRewriting:
    """COUNT(DISTINCT ?x) is emitted by this repo's own bridge generator."""

    DISTINCT = "SELECT (COUNT(DISTINCT ?answer) AS ?count) WHERE { wd:Q1 wdt:P2 ?answer . }"
    PLAIN = "SELECT (COUNT(?object) AS ?count) WHERE { wd:Q1 wdt:P2 ?object . }"

    def test_distinct_is_rewritable(self):
        out = sparql.count_to_select(self.DISTINCT)
        assert out is not None and "SELECT DISTINCT ?answer" in out

    def test_distinct_label_select(self):
        out = sparql.to_label_select(self.DISTINCT)
        assert out is not None and "?answerLabel" in out

    def test_plain_count_still_rewritable(self):
        assert "SELECT DISTINCT ?object" in sparql.count_to_select(self.PLAIN)


class TestSparqlCountRaisesOnBadPayload:
    """A 200 response that is not a result set is not 'zero answers'."""

    def test_error_payload_raises(self):
        class Fake(sparql.SparqlClient):
            def __init__(self):
                self.endpoint = "http://fake"
                self.session = None

            def query(self, q):
                return {"error": "backend unavailable"}

        with pytest.raises(sparql.SparqlError, match="no SPARQL result set"):
            Fake().count("SELECT (COUNT(?x) AS ?count) WHERE {}")

    def test_genuine_empty_result_is_zero(self):
        class Fake(sparql.SparqlClient):
            def __init__(self):
                self.endpoint = "http://fake"
                self.session = None

            def query(self, q):
                return {"results": {"bindings": []}}

        assert Fake().count("SELECT (COUNT(?x) AS ?count) WHERE {}") == 0


class TestIdentifierValidation:
    @pytest.mark.parametrize("value", ["Q42", "Q1", "Q123456"])
    def test_valid_qids(self, value):
        assert filters.is_qid(value) and filters.require_qid(value) == value

    @pytest.mark.parametrize("value", ["Q0", "42", "Q42x", "", "P31", "Q1 . ?x ?y ?z ."])
    def test_invalid_qids(self, value):
        assert not filters.is_qid(value)
        with pytest.raises(filters.InvalidIdentifier):
            filters.require_qid(value)

    def test_injection_attempt_is_rejected(self):
        with pytest.raises(filters.InvalidIdentifier, match="not a Wikidata entity id"):
            filters.require_qid("Q1 } INSERT DATA { ")

    def test_pid_validation(self):
        assert filters.is_pid("P31") and not filters.is_pid("Q31")


class TestExtractorDedupKey:
    """Dedup was keyed on entity id alone, dropping every later property."""

    HEADER_ROWS = [
        ["Q42", "Ada", "P19", "place of birth", "wikibase-item", "Q84", "Q90", "Paris", "updated", "t", "u"],
        ["Q42", "Ada", "P17", "country", "wikibase-item", "NEW", "Q30", "US", "created", "t", "u"],
        ["Q42", "Ada", "P106", "occupation", "wikibase-item", "NEW", "Q901", "scientist", "created", "t", "u"],
        ["Q42", "Ada", "P19", "place of birth", "wikibase-item", "Q84", "Q90", "Paris", "updated", "t", "u"],
        ["Q7", "Bob", "P17", "country", "wikibase-item", "NEW", "Q30", "US", "created", "t", "u"],
    ]

    def test_multiple_properties_per_entity_survive(self, tmp_path):
        ext = _load("scripts/extract_triple_changes.py")
        out = tmp_path / "t.csv"
        sink = ext.CsvSink(out, resume=False)
        added = sink.write_rows(self.HEADER_ROWS)
        sink.close()
        assert added == 4, "the true duplicate should be dropped, the rest kept"
        rows = list(csv.reader(out.open()))[1:]
        q42 = [r for r in rows if r[0] == "Q42"]
        assert len(q42) == 3, "Level 2 needs several attributes for one subject"
        assert {r[2] for r in q42} == {"P19", "P17", "P106"}

    def test_resume_does_not_discard_new_properties(self, tmp_path):
        ext = _load("scripts/extract_triple_changes.py")
        out = tmp_path / "t.csv"
        sink = ext.CsvSink(out, resume=False)
        sink.write_rows(self.HEADER_ROWS)
        sink.close()

        sink2 = ext.CsvSink(out, resume=True)
        added = sink2.write_rows([
            ["Q42", "Ada", "P569", "date of birth", "time", "NEW", "1815", "1815", "created", "t", "u"],
            ["Q42", "Ada", "P17", "country", "wikibase-item", "NEW", "Q30", "US", "created", "t", "u"],
        ])
        sink2.close()
        assert added == 1, "a new property is new; an already-written one is not"


class TestIsCorrectConsistency:
    """oracle/wiki_corpus stored containment while DA/CoT/RAG stored exact match."""

    def test_all_runners_derive_is_correct_from_exact_match(self):
        for script in ("scripts/eval/DA.py", "scripts/eval/CoT.py", "scripts/eval/RAG.py",
                       "scripts/eval/oracle.py", "scripts/eval/wiki_corpus.py"):
            text = (ROOT / script).read_text()
            assert '"is_correct": bool(scores["contains_match"])' not in text, script
