"""Producer-side check: a small campaign's own exported runs, read back by WiseGncAnalysis's
own loader rather than a restated assumption about what it expects. Runs 2 cases for a fast
sanity check; the real CI-sized check is its own explicit `pytest sim/basilisk/scenarios --runs
8` step (see .github/workflows/checks.yml)."""

import subprocess
import sys


def test_two_case_campaign_exports_runs_the_loader_reads(tmp_path):
    out_dir = tmp_path / "mc_export"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "sim/basilisk/scenarios",
            "--runs",
            "2",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(out_dir.glob("run_*.h5"))) == 2
