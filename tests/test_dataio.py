"""Dataset loading must accept every shape the released files actually use."""

import json

import pytest

from livesearchbench import dataio

RELEASED = ["demo.json", "bench/2021/level1.json", "bench/2021/level2.json",
            "bench/2021/level3.json", "bench/2025/level1.json",
            "bench/2025/level2.json", "bench/2025/level3.json"]


class TestLoadReleasedFiles:
    @pytest.mark.parametrize("path", RELEASED)
    def test_every_released_split_loads(self, path):
        items, meta = dataio.load_instances(path)
        assert items and meta["dataset_path"] == path
        assert all("question" in i and "answer" in i for i in items)

    def test_expected_counts(self):
        assert len(dataio.load_instances("demo.json")[0]) == 30
        for year in ("2021", "2025"):
            for level, n in (("level1", 150), ("level2", 100), ("level3", 50)):
                items, _ = dataio.load_instances(f"bench/{year}/{level}.json")
                assert len(items) == n, f"{year}/{level}"

    def test_bare_list_shape(self):
        # demo.json has no wrapper object at all.
        assert isinstance(json.loads(open("demo.json").read()), list)
        assert len(dataio.load_instances("demo.json")[0]) == 30

    def test_dataset_info_wrapper_shape(self):
        # bench/2025/level3.json uses "dataset_info" where the others use "metadata".
        raw = json.loads(open("bench/2025/level3.json").read())
        assert "dataset_info" in raw and "metadata" not in raw
        items, meta = dataio.load_instances("bench/2025/level3.json")
        assert len(items) == 50 and meta["level"] == "level3"


class TestInference:
    def test_level_and_year_from_path_and_items(self):
        _, meta = dataio.load_instances("bench/2025/level1.json")
        assert meta["level"] == "level1" and meta["year"] == "2025"

    def test_mixed_split_infers_nothing(self):
        # demo.json mixes levels and years, so neither can be inferred.
        _, meta = dataio.load_instances("demo.json")
        assert meta.get("level") is None and meta.get("year") is None

    def test_year_is_not_taken_from_a_filename(self):
        # A model name or timestamp in the basename must not be read as a year.
        assert dataio.infer_year("outputs/level1_gpt_4o_20250913.json", []) is None


class TestErrors:
    def test_missing_file_names_the_real_locations(self, tmp_path):
        with pytest.raises(dataio.DatasetFormatError) as exc:
            dataio.load_instances(tmp_path / "nope.json")
        assert "bench/<year>/level<N>.json" in str(exc.value)

    def test_object_without_an_instance_list(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"metadata": {}, "rows": []}))
        with pytest.raises(dataio.DatasetFormatError) as exc:
            dataio.load_instances(p)
        assert "qa_pairs" in str(exc.value)

    def test_items_missing_required_fields(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"question": "q"}]))
        with pytest.raises(dataio.DatasetFormatError) as exc:
            dataio.load_instances(p)
        assert "answer" in str(exc.value)

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        with pytest.raises(dataio.DatasetFormatError):
            dataio.load_instances(p)


class TestRoundTrip:
    def test_save_run_then_load_results(self, tmp_path):
        results = [{"question": "q", "expected_answer": "a", "model_answer": "a", "is_correct": True}]
        paths = dataio.save_run(
            results=results, summary={"accuracy": 100.0, "max_search_calls_allowed": 5},
            method="RAG", model_name="qwen-2.5/14b", data_path="bench/2025/level1.json",
            output_dir=str(tmp_path), metadata={"level": "level1", "year": "2025"},
        )
        # The search budget must survive into the filename for summarize_rag_budget.py.
        assert "_maxiter_5_" in paths["results"]
        loaded, meta = dataio.load_results(paths["results"])
        assert loaded == results and meta["method"] == "RAG"
        assert "/" not in paths["results"].rsplit("_maxiter", 1)[0].rsplit("/", 1)[-1]

    def test_load_results_accepts_the_legacy_bare_list(self, tmp_path):
        p = tmp_path / "old.json"
        p.write_text(json.dumps([{"question": "q", "is_correct": False}]))
        items, _ = dataio.load_results(p)
        assert len(items) == 1

    def test_ensure_parent_creates_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.csv"
        dataio.ensure_parent(target)
        assert target.parent.is_dir()


class TestNormalizeInstance:
    def test_fills_in_optional_fields(self):
        out = dataio.normalize_instance({"question": " q ", "answer": "a"})
        assert out["question"] == "q"
        assert out["sparql_verification"] == "" and out["answer_aliases"] == []


class TestSaveRunMetadataPrecedence:
    """A run's own summary must never be overwritten by the dataset header.

    Every released split defines ``total_questions`` in its metadata block, so
    passing that block through as ``metadata=`` used to make a ``--limit 25``
    run report ``total_questions: 150``.
    """

    def test_summary_wins_over_dataset_metadata(self, tmp_path):
        import json as _json

        dataset_header = {"total_questions": 150, "level": "level1", "year": "2025"}
        paths = dataio.save_run(
            results=[{"question": "q", "expected_answer": "a", "model_answer": "a"}],
            summary={"total_questions": 25, "accuracy": 88.0},
            method="DA", model_name="m", data_path="bench/2025/level1.json",
            output_dir=str(tmp_path), metadata=dataset_header,
        )
        summary = _json.loads(open(paths["summary"]).read())
        assert summary["total_questions"] == 25, "the run's own count must survive"
        assert summary["accuracy"] == 88.0
        # Fields the dataset header contributes and the summary does not are kept.
        assert summary["year"] == "2025" and summary["method"] == "DA"

    def test_run_metadata_is_not_shadowed_either(self, tmp_path):
        import json as _json

        paths = dataio.save_run(
            results=[{"question": "q", "expected_answer": "a", "model_answer": "a"}],
            summary={}, method="RAG", model_name="real-model",
            data_path="bench/2025/level1.json", output_dir=str(tmp_path),
            metadata={"model": "some-other-model-from-the-header"},
        )
        summary = _json.loads(open(paths["summary"]).read())
        assert summary["model"] == "real-model"
