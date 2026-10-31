"""Producer-side conformance: running the scenario's own pytest campaign at a small scale and
confirming it succeeds, which includes its own `pytest_sessionfinish` call into WiseGncAnalysis's
real `contract.v0.check_campaign` -- not a restated assumption about the columns it writes."""

import json
import subprocess
import sys


def test_eight_run_campaign_passes_its_own_contract_check(tmp_path):
    out_dir = tmp_path / "mc_contract"
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

    manifest = json.loads((out_dir / "manifest.json").read_text())
    versions = manifest["versions"]
    assert set(versions) >= {"bsk", "gnc-algorithms", "wise-dynamics"}
    assert all(versions[key] for key in ("bsk", "gnc-algorithms", "wise-dynamics"))
