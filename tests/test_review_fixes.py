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


class TestDumpDeltaValueSets:
    """The T0 index kept one value per (s, p), hiding real changes."""

    @staticmethod
    def _run(tmp_path, extra=()):
        import subprocess
        fixture = ROOT / "data" / "sample"
        out = tmp_path / "delta.csv"
        cmd = [sys.executable, str(ROOT / "scripts" / "extract_dump_delta.py"),
               "--t0", str(fixture / "dump_T0.json.gz"),
               "--t1", str(fixture / "dump_T1.json.gz"),
               "--output", str(out), "--offline", *extra]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode == 0, proc.stderr
        with out.open() as handle:
            return {(r["property_id"], r["old_value"], r["new_value"]): r
                    for r in csv.DictReader(handle)}

    def test_multivalued_change_is_detected(self, tmp_path):
        """{Q901, Q902} -> {Q901, Q903}: picking one value saw no change."""
        rows = self._run(tmp_path, ["--no-allowlist"])
        p106 = [k for k in rows if k[0] == "P106"]
        assert p106, "the changed value of a multivalued property must appear"
        assert p106[0][2] == "Q903"
        assert "Q902" in p106[0][1]

    def test_unchanged_value_of_a_multivalued_property_is_not_emitted(self, tmp_path):
        rows = self._run(tmp_path, ["--no-allowlist"])
        assert not any(k[0] == "P106" and k[2] == "Q901" for k in rows), \
            "Q901 is in both snapshots and is not a change"

    def test_unit_change_is_detected(self, tmp_path):
        """Same amount, metres -> kilometres. The amount alone looked equal."""
        rows = self._run(tmp_path, ["--no-allowlist"])
        p2046 = [k for k in rows if k[0] == "P2046"]
        assert p2046, "a unit change is a change"
        assert "Q11573" in p2046[0][1] and "Q828224" in p2046[0][2]

    def test_time_precision_change_is_detected(self, tmp_path):
        """Same instant, day -> year precision. The timestamp alone looked equal."""
        rows = self._run(tmp_path, ["--no-allowlist"])
        p569 = [k for k in rows if k[0] == "P569"]
        assert p569, "a precision change is a change"
        assert "p11" in p569[0][1] and "p9" in p569[0][2]

    def test_entity_values_remain_bare_qids(self, tmp_path):
        """Downstream generators match new_value against ^Q\\d+$."""
        rows = self._run(tmp_path)
        for (pid, _old, new), row in rows.items():
            if row["property_type"] == "wikibase-item":
                assert filters.is_qid(new), f"{pid} emitted {new!r}"

    def test_allowlist_still_filters(self, tmp_path):
        """occupation/area/date of birth are not on the curated list."""
        with_list = self._run(tmp_path / "a")
        without = self._run(tmp_path / "b", ["--no-allowlist"])
        assert len(with_list) < len(without)
        assert {k[0] for k in with_list} <= {"P176", "P19", "P54"}


class TestCanonicalValue:
    """Structured values must not be truncated to their headline field."""

    @staticmethod
    def _canon(snak):
        module = _load("scripts/extract_dump_delta.py")
        return module.canonical_value(snak)

    def _quantity(self, amount, unit):
        return {"snaktype": "value", "datatype": "quantity",
                "datavalue": {"type": "quantity",
                              "value": {"amount": amount,
                                        "unit": f"http://www.wikidata.org/entity/{unit}"}}}

    def test_same_amount_different_unit_differs(self):
        assert self._canon(self._quantity("+100", "Q11573")) != \
               self._canon(self._quantity("+100", "Q828224"))

    def test_same_amount_same_unit_matches(self):
        assert self._canon(self._quantity("+100", "Q11573")) == \
               self._canon(self._quantity("+100", "Q11573"))

    def test_time_precision_is_significant(self):
        def t(precision):
            return {"snaktype": "value", "datatype": "time",
                    "datavalue": {"type": "time",
                                  "value": {"time": "+1815-12-10T00:00:00Z",
                                            "precision": precision,
                                            "calendarmodel": "http://www.wikidata.org/entity/Q1985727"}}}
        assert self._canon(t(11)) != self._canon(t(9))

    def test_coordinate_globe_is_significant(self):
        def c(globe):
            return {"snaktype": "value", "datatype": "globe-coordinate",
                    "datavalue": {"type": "globecoordinate",
                                  "value": {"latitude": 1.0, "longitude": 2.0,
                                            "precision": 0.1,
                                            "globe": f"http://www.wikidata.org/entity/{globe}"}}}
        assert self._canon(c("Q2")) != self._canon(c("Q111"))

    def test_novalue_snaks_are_skipped(self):
        assert self._canon({"snaktype": "novalue", "datatype": "quantity"}) is None


class TestGeneratorFailsLoud:
    """A permanent model error produced an empty dataset and exit 0."""

    def test_permanent_http_errors_abort(self):
        for script in ("scripts/generate_level1.py", "scripts/generate_level2.py",
                       "scripts/generate_level3.py"):
            text = (ROOT / script).read_text()
            assert "401, 403, 404" in text, f"{script} still swallows auth failures"
            assert "raise SystemExit(" in text

    def test_empty_output_is_not_success(self):
        for script in ("scripts/generate_level1.py", "scripts/generate_level2.py",
                       "scripts/generate_level3.py"):
            text = (ROOT / script).read_text()
            assert "No questions were generated" in text, script


class TestResumeIsolation:
    """Resume matched on level/method/model only, ignoring the real settings."""

    @staticmethod
    def _path(**over):
        module = _load("scripts/eval/DA.py")
        base = dict(dataset="demo.json", model="m", temperature=0.0,
                    n_samples=1, max_tokens=256, method="DA")
        base.update(over)
        fp = module.run_fingerprint(**base)
        return module.default_partial_path(
            output_dir="/tmp", meta={"level": "level1", "year": "2025"},
            model_name="m", fingerprint=fp).name

    def test_identical_settings_share_a_sidecar(self):
        assert self._path() == self._path()

    @pytest.mark.parametrize("field,value", [
        ("temperature", 0.9), ("n_samples", 4),
        ("dataset", "bench/2025/level1.json"), ("max_tokens", 512),
    ])
    def test_changed_setting_gets_its_own_sidecar(self, field, value):
        assert self._path() != self._path(**{field: value}), \
            f"changing {field} must not reuse the previous run's answers"

    def test_fingerprint_is_order_independent(self):
        module = _load("scripts/eval/DA.py")
        assert module.run_fingerprint(a=1, b=2) == module.run_fingerprint(b=2, a=1)


class TestPassAtKDoesNotPoolAcrossRuns:
    """The combined report merged generations from different models."""

    def test_one_sample_per_run_cannot_yield_pass_at_2(self, tmp_path):
        import json
        import subprocess

        for model, answer in (("modelA", "Paris"), ("modelB", "Berlin")):
            (tmp_path / f"{model}_results.json").write_text(json.dumps({
                "metadata": {"method": "DA", "model": model},
                "results": [{"question": "Capital of France?", "expected_answer": "Paris",
                             "model_answer": answer, "is_correct": answer == "Paris",
                             "level": 1}],
            }))
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts/analysis/score.py"),
             *[str(p) for p in sorted(tmp_path.glob("*_results.json"))],
             "--pass-at-k", "2", "--format", "json"],
            capture_output=True, text=True, cwd=ROOT)
        assert proc.returncode == 0, proc.stderr
        report = json.loads(proc.stdout)
        pk = report["combined"]["scores"]["pass_at_k"]
        assert pk["n_runs"] == 2
        assert pk["n_questions"] == 2, "one group per (run, question)"
        assert pk["values"]["exact_match"]["2"]["value"] is None
        assert pk["values"]["exact_match"]["2"]["skipped"] == 2
