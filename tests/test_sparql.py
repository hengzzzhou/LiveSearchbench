"""SPARQL query rewriting. No network access; live checks live in the smoke test."""

import json

import pytest

from livesearchbench import sparql

COUNT_QUERY = "SELECT (COUNT(?object) AS ?count) WHERE {\n  wd:Q4810517 wdt:P4552 ?object .\n}"


class TestCountToSelect:
    def test_rewrites_the_projection(self):
        out = sparql.count_to_select(COUNT_QUERY)
        assert out.startswith("SELECT DISTINCT ?object")
        assert "COUNT" not in out.upper()

    def test_preserves_the_where_clause(self):
        assert "wd:Q4810517 wdt:P4552 ?object" in sparql.count_to_select(COUNT_QUERY)

    def test_returns_none_for_a_plain_select(self):
        assert sparql.count_to_select("SELECT ?x WHERE { ?x ?y ?z }") is None

    def test_handles_whitespace_and_case_variation(self):
        q = "select ( count( ?entity )   AS ?c ) where { ?entity wdt:P31 wd:Q5 . }"
        out = sparql.count_to_select(q)
        assert out is not None
        assert "SELECT DISTINCT ?entity" in out
        assert "count(" not in out.lower()

    def test_tolerates_empty_input(self):
        assert sparql.count_to_select("") is None
        assert sparql.count_to_select(None) is None


class TestToLabelSelect:
    def test_projects_both_the_variable_and_its_label(self):
        out = sparql.to_label_select(COUNT_QUERY)
        assert "?object" in out and "?objectLabel" in out

    def test_attaches_the_label_service(self):
        out = sparql.to_label_select(COUNT_QUERY)
        assert "SERVICE wikibase:label" in out and 'wikibase:language "en"' in out

    def test_language_is_configurable(self):
        assert 'wikibase:language "zh"' in sparql.to_label_select(COUNT_QUERY, lang="zh")

    def test_returns_none_for_a_plain_select(self):
        assert sparql.to_label_select("SELECT ?x WHERE { ?x ?y ?z }") is None


class TestWithLabels:
    def test_is_idempotent(self):
        once = sparql.with_labels("SELECT ?x WHERE { ?x wdt:P31 wd:Q5 . }")
        assert sparql.with_labels(once) == once

    def test_inserts_before_the_closing_brace(self):
        out = sparql.with_labels("SELECT ?x WHERE { ?x wdt:P31 wd:Q5 . }")
        assert out.rstrip().endswith("}")
        assert out.index("SERVICE") < out.rindex("}")

    def test_leaves_a_malformed_query_alone(self):
        assert sparql.with_labels("not a query") == "not a query"


class TestEveryReleasedProgramIsRewritable:
    """Every shipped verification program must survive the rewrite."""

    @pytest.mark.parametrize("path", [
        "demo.json", "bench/2021/level1.json", "bench/2021/level2.json",
        "bench/2025/level1.json", "bench/2025/level2.json",
    ])
    def test_rewrite_succeeds(self, path):
        raw = json.loads(open(path).read())
        items = raw if isinstance(raw, list) else raw.get("qa_pairs", raw.get("dataset_info", []))
        programs = [i.get("sparql_verification", "") for i in items if i.get("sparql_verification")]
        assert programs, f"{path} has no verification programs"
        rewritten = [sparql.to_label_select(p) for p in programs]
        ok = [r for r in rewritten if r]
        # The overwhelming majority are COUNT programs and must rewrite cleanly.
        assert len(ok) / len(programs) > 0.9, f"only {len(ok)}/{len(programs)} rewrote in {path}"
