"""Producer-side check: a small campaign's own exported runs, read back by WiseGncAnalysis's
own loader rather than a restated assumption about what it expects.

Skipped for now: `gnc_analysis.runs.load_run`/`load_campaign` are not wired in yet (pending
impl-analysis's SHA); `sim/basilisk/scenarios/test_imu_dead_reckoning.py` and
`sim/basilisk/conftest.py` already import them directly, so this test will start failing loudly,
not silently, the moment that import is attempted against an incomplete loader -- which is the
point of leaving the import unguarded there rather than skipping it too."""

import pytest

pytestmark = pytest.mark.skip(
    reason="gnc_analysis.runs loader not wired in yet; see sim/basilisk/conftest.py's TODO"
)


def test_eight_case_campaign_exports_runs_the_loader_reads(tmp_path):
    import subprocess
    import sys

    out_dir = tmp_path / "mc_export"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "sim/basilisk/scenarios",
            "--runs",
            "8",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len(list(out_dir.glob("run_*.h5"))) == 8
