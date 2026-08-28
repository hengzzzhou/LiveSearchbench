"""Metric behaviour, including the exact-match vs containment distinction."""

import pytest

from livesearchbench import scoring


class TestNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("The Bowers Mountains.", "bowers mountains"),
        ("  COVID-19  ", "covid 19"),
        ("a Saudi Arabian club", "saudi arabian club"),
        ("", ""),
        (None, ""),
    ])
    def test_normalize(self, raw, expected):
        assert scoring.normalize_answer(raw) == expected


class TestExactMatch:
    def test_identical(self):
        assert scoring.exact_match("Naushad", "Naushad")

    def test_ignores_case_articles_and_punctuation(self):
        assert scoring.exact_match("the bowers mountains.", "Bowers Mountains")

    def test_rejects_a_sentence_containing_the_answer(self):
        # This is the whole point: containment accepts it, exact match must not.
        verbose = "It is located in the Bowers Mountains of Antarctica."
        assert scoring.contains_match(verbose, "Bowers Mountains")
        assert not scoring.exact_match(verbose, "Bowers Mountains")

    def test_wrong_answer(self):
        assert not scoring.exact_match("Vladimir Kurnev", "Anatoly Baidachny")

    def test_empty_prediction_is_never_correct(self):
        assert not scoring.exact_match("", "Naushad")
        assert not scoring.contains_match("", "Naushad")

    def test_aliases(self):
        assert scoring.exact_match("USA", "United States", aliases=["USA", "US"])
        assert not scoring.exact_match("Canada", "United States", aliases=["USA", "US"])

    def test_gold_may_be_a_list(self):
        assert scoring.exact_match("Antarctica", ["Antarctica", "the Antarctic"])


class TestTokenF1:
    def test_perfect(self):
        assert scoring.token_f1("Bowers Mountains", "Bowers Mountains") == 1.0

    def test_no_overlap(self):
        assert scoring.token_f1("Naushad", "Antarctica") == 0.0

    def test_partial_overlap_is_between(self):
        f1 = scoring.token_f1("Bowers Mountains of Antarctica", "Bowers Mountains")
        assert 0.0 < f1 < 1.0

    def test_f1_never_below_exact_match(self):
        for pred, gold in [("a", "a"), ("a b", "a"), ("x", "y")]:
            assert scoring.token_f1(pred, gold) >= float(scoring.exact_match(pred, gold))


class TestBootstrap:
    def test_deterministic_given_a_seed(self):
        vals = [100.0] * 30 + [0.0] * 70
        a = scoring.bootstrap_ci(vals, resamples=500, seed=7)
        b = scoring.bootstrap_ci(vals, resamples=500, seed=7)
        assert a == b

    def test_interval_brackets_the_mean(self):
        ci = scoring.bootstrap_ci([100.0] * 30 + [0.0] * 70, resamples=2000, seed=0)
        assert ci["mean"] == pytest.approx(30.0)
        assert ci["lo"] < ci["mean"] < ci["hi"]
        # Binomial 95% CI for 30/100 is roughly [21, 40].
        assert 15 < ci["lo"] < 30 < ci["hi"] < 45

    def test_degenerate_inputs(self):
        assert scoring.bootstrap_ci([])["n"] == 0
        single = scoring.bootstrap_ci([42.0])
        assert single["lo"] == single["hi"] == 42.0


class TestPassAtK:
    def test_k_one_equals_success_rate(self):
        assert scoring.pass_at_k(4, 1, 1) == pytest.approx(0.25)

    def test_certain_when_k_covers_all_failures(self):
        assert scoring.pass_at_k(4, 1, 4) == 1.0

    def test_zero_correct(self):
        assert scoring.pass_at_k(8, 0, 3) == 0.0

    def test_monotonic_in_k(self):
        vals = [scoring.pass_at_k(16, 3, k) for k in range(1, 9)]
        assert vals == sorted(vals)


class TestAggregate:
    def test_reports_all_three_metrics_with_intervals(self):
        items = [
            {"model_answer": "Naushad", "expected_answer": "Naushad", "level": 1},
            {"model_answer": "The answer is Naushad.", "expected_answer": "Naushad", "level": 1},
            {"model_answer": "wrong", "expected_answer": "Antarctica", "level": 2},
        ]
        out = scoring.aggregate(items, resamples=200, seed=0)
        overall = out["overall"]
        assert overall["n"] == 3
        # One exact match out of three, two containments out of three.
        assert overall["exact_match"]["value"] == pytest.approx(100 / 3, abs=0.01)
        assert overall["contains_match"]["value"] == pytest.approx(200 / 3, abs=0.01)
        assert overall["contains_match"]["value"] > overall["exact_match"]["value"]
        for metric in ("exact_match", "token_f1", "contains_match"):
            assert overall[metric]["ci_low"] <= overall[metric]["value"] <= overall[metric]["ci_high"]

    def test_groups_by_level(self):
        items = [
            {"model_answer": "Naushad", "expected_answer": "Naushad", "level": 1},
            {"model_answer": "Brazil", "expected_answer": "Antarctica", "level": 2},
        ]
        out = scoring.aggregate(items, resamples=100, seed=0)
        assert set(out["by_level"]) == {"1", "2"}
        assert out["by_level"]["1"]["exact_match"]["value"] == 100.0
        assert out["by_level"]["2"]["exact_match"]["value"] == 0.0


class TestNormalisationEdgeCases:
    """Answers that consist only of stop words normalise to the empty string.

    This is inherent to SQuAD-style normalisation. No Wikidata entity label is
    a bare article, so it does not affect the benchmark, but the behaviour is
    pinned here so a future change to normalize_answer is a deliberate one.
    """

    def test_bare_article_normalises_to_empty(self):
        assert scoring.normalize_answer("the") == ""

    def test_empty_normalisation_is_not_a_match(self):
        assert not scoring.exact_match("the", "a")

    def test_multiword_names_containing_articles_still_match(self):
        assert scoring.exact_match("The Who", "the who")
        assert scoring.exact_match("A Tale of Two Cities", "Tale of Two Cities")
