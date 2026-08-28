"""Candidate filtering, including the relation list recovered from git history."""

from livesearchbench import filters


class TestRelationLists:
    def test_allowlist_restored_intact(self):
        # 198 labels, recovered from filtered.json which commit c404540 deleted.
        assert len(filters.relation_allowlist()) == 198

    def test_paper_appendix_predicates_are_denied(self):
        for pid in ("P18", "P31", "P279", "P373", "P1889"):
            assert not filters.is_allowed_relation(pid, "anything")

    def test_curated_relations_are_allowed(self):
        for label in ("winner", "eye color", "place of burial"):
            assert filters.is_allowed_relation("P1346", label)

    def test_relation_outside_the_allowlist_is_rejected(self):
        assert not filters.is_allowed_relation("P9999", "some obscure identifier")

    def test_allowlist_can_be_disabled_but_denylist_cannot(self):
        assert filters.is_allowed_relation("P9999", "obscure", use_allowlist=False)
        assert not filters.is_allowed_relation("P18", "image", use_allowlist=False)

    def test_label_matching_is_case_insensitive(self):
        assert filters.is_allowed_relation("P1346", "  WINNER  ")


class TestEntityQuality:
    def test_accepts_a_normal_entity(self):
        ok, reason = filters.is_allowed_entity(
            label="Bowers Mountains", sitelinks={"enwiki": {"title": "Bowers Mountains"}})
        assert ok and reason == ""

    def test_requires_a_label(self):
        ok, reason = filters.is_allowed_entity(label="", sitelinks={"enwiki": {}})
        assert not ok and "label" in reason

    def test_requires_an_english_sitelink_by_default(self):
        ok, reason = filters.is_allowed_entity(label="X", sitelinks={"dewiki": {}})
        assert not ok and "English Wikipedia" in reason
        ok, _ = filters.is_allowed_entity(label="X", sitelinks={"dewiki": {}}, require_enwiki=False)
        assert ok

    def test_empty_sitelink_value_still_counts_as_present(self):
        ok, _ = filters.is_allowed_entity(label="X", sitelinks={"enwiki": {}})
        assert ok

    def test_rejects_disambiguation_and_list_pages(self):
        for qid in ("Q4167410", "Q13406463"):
            ok, reason = filters.is_allowed_entity(
                label="Mercury", sitelinks={"enwiki": {}}, instance_of=[qid])
            assert not ok and "instance of" in reason

    def test_rejects_wiki_namespace_surface_forms(self):
        ok, reason = filters.is_allowed_entity(
            label="Category:Living people", sitelinks={"enwiki": {}})
        assert not ok and "category:" in reason


class TestStatementValidity:
    def test_prefers_preferred_rank(self):
        best = filters.best_statement([{"rank": "normal", "v": 1}, {"rank": "preferred", "v": 2}])
        assert best["v"] == 2

    def test_drops_deprecated(self):
        assert filters.best_statement([{"rank": "deprecated", "v": 1}]) is None

    def test_falls_back_to_normal(self):
        assert filters.best_statement([{"rank": "deprecated"}, {"rank": "normal", "v": 3}])["v"] == 3

    def test_missing_rank_treated_as_normal(self):
        assert filters.best_statement([{"v": 1}])["v"] == 1

    def test_dedup_key_prefers_statement_id(self):
        assert filters.dedup_key("Q1", "P2", "Q1$abc").startswith("stmt:")
        assert filters.dedup_key("q1", "p2") == "sr:Q1|P2"


class TestFilterStats:
    def test_funnel_percentages_match_the_papers_table(self):
        stats = filters.FilterStats()
        stats.stage("Raw delta triples", 823461)
        stats.stage("Relation allow-list", 142538)
        stats.stage("Entity quality", 76819)
        funnel = stats.to_dict()["funnel"]
        assert funnel[1]["survival_vs_previous"] == 17.31
        assert funnel[2]["survival_vs_previous"] == 53.89

    def test_drop_reasons_are_counted_and_sorted(self):
        stats = filters.FilterStats()
        stats.drop("no enwiki", 5)
        stats.drop("deprecated", 2)
        stats.drop("no enwiki")
        reasons = stats.to_dict()["drop_reasons"]
        assert list(reasons) == ["no enwiki", "deprecated"] and reasons["no enwiki"] == 6

    def test_render_produces_markdown(self):
        stats = filters.FilterStats()
        stats.stage("Raw", 100)
        assert "| Pipeline Stage |" in stats.render()
