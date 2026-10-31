"""Checks `sim.basilisk.pytest_plugin`: its own provenance helper reads this package's installed
version rather than running git, and a 2-case campaign run through its fixtures, with a stub
flight-software module, writes one run file per case."""

import textwrap
from importlib.metadata import version as pkg_version

from sim.basilisk.pytest_plugin import _wise_dynamics_version


def test_version_matches_installed_distribution_metadata():
    reported = _wise_dynamics_version()
    assert reported.startswith(pkg_version("wise-dynamics"))


def test_two_case_campaign_runs_and_writes_one_file_per_case(pytester):
    pytester.makeconftest('pytest_plugins = ["sim.basilisk.pytest_plugin"]')
    pytester.makepyfile(
        test_campaign=textwrap.dedent(
            """
            import numpy as np
            from Basilisk.architecture import messaging, sysModel

            from sim.basilisk.run import run_case


            class _StubFsw(sysModel.SysModel):
                def __init__(self):
                    super().__init__()
                    self.output_msg = messaging.NavAttMsg()

                def UpdateState(self, CurrentSimNanos):
                    payload = messaging.NavAttMsgPayload()
                    payload.sigma_BN = [0.1, 0.2, 0.3]
                    self.output_msg.write(payload, CurrentSimNanos, self.moduleID)


            def test_case(case_index, plant, run_writer):
                stub = _StubFsw()
                stub.ModelTag = "stub"
                recorders = run_case(
                    plant, stub, {"stub_out": stub.output_msg}, duration_s=2.0, record_s=1.0
                )
                path = run_writer({"sigma_BN": (np.array(recorders["truth"].sigma_BN), "-")})
                assert path.exists()
            """
        )
    )

    out_dir = pytester.path / "out"
    result = pytester.runpytest_subprocess("--runs", "2", "--out", str(out_dir))

    result.assert_outcomes(passed=2)
    written = sorted(out_dir.glob("run_*.h5"))
    assert len(written) == 2
