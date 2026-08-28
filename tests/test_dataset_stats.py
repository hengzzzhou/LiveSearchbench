"""The dataset description must stay derived from the benchmark files."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "analysis" / "dataset_stats.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=ROOT, capture_output=True, text=True)


class TestDatasetStats:
    def test_description_matches_the_data(self):
        result = _run("--check")
        assert result.returncode == 0, result.stderr

    def test_counts_match_the_released_files(self):
        stats = json.loads((ROOT / "data_description.json").read_text())
        assert stats["total_qa_pairs"] == 600
        assert stats["level_distribution"] == {"L1": 300, "L2": 200, "L3": 100}
        assert len(stats["splits"]) == 6

    def test_every_instance_carries_a_verification_program(self):
        stats = json.loads((ROOT / "data_description.json").read_text())
        assert stats["instances_with_sparql_verification"] == stats["total_qa_pairs"]

    def test_check_detects_a_stale_description(self, tmp_path):
        stale = tmp_path / "stale.json"
        stale.write_text(json.dumps({"total_qa_pairs": 999}))
        result = _run("--check", "--output", str(stale))
        assert result.returncode == 1
        assert "STALE" in result.stderr and "999" in result.stderr

    def test_description_is_marked_as_generated(self):
        stats = json.loads((ROOT / "data_description.json").read_text())
        assert stats["_generated_by"].endswith("dataset_stats.py")
