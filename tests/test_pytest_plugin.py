"""Checks `sim.basilisk.pytest_plugin`: its own provenance helper reads this package's installed
version from distribution metadata; a 2-case campaign run through its fixtures, with a stub
flight-software module, writes one run file per case, each carrying its own draws and its own
distinct sensor noise; a campaign fanned over workers shares one output directory; a campaign
still runs where pytest-xdist is absent; and a session with no case creates no directory at
all."""

import textwrap
from importlib.metadata import version as pkg_version

import h5py
import numpy as np

from sim.basilisk.pytest_plugin import _wise_dynamics_version

_CAMPAIGN_SOURCE = textwrap.dedent(
    """
    import numpy as np
    import pytest
    from Basilisk.architecture import messaging, sysModel

    from sim.basilisk.run import run_case


    @pytest.fixture
    def draws():
        # Zero draws for every case, overriding the plugin's own dispersed `draws` fixture: the
        # only thing left free to differ between cases is each one's own IMU noise seed, so a
        # difference in recorded `gyro` below traces to that seed alone.
        return {
            "sigma0_BN": np.zeros(3),
            "omega0_BN_B": np.zeros(3),
            "gyro_bias": np.zeros(3),
            "accel_bias": np.zeros(3),
        }


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
        path = run_writer(
            {
                "sigma_BN": (np.array(recorders["truth"].sigma_BN), "-"),
                "gyro": (np.array(recorders["imu"].AngVelPlatform), "rad/s"),
            }
        )
        assert path.exists()
    """
)


def test_version_matches_installed_distribution_metadata():
    reported = _wise_dynamics_version()
    assert reported.startswith(pkg_version("wise-dynamics"))


def test_two_case_campaign_runs_with_distinct_noise_and_recorded_draws(pytester):
    """Each case's own run file carries different IMU noise and the draws the plugin dispersed
    for that case. The campaign's own `draws` fixture is overridden to zero so every case starts
    identical and tumble-free; `gyro` then isolates the IMU noise seed alone (R1: a shared seed
    across cases would make it identical), and the `draw_*` attrs are checked regardless (R5: a
    writer that drops them would leave these attrs out). Relies on the `pytest11` entry point
    to load the plugin in the subprocess -- no `conftest.py` declaration is written here."""
    pytester.makepyfile(test_campaign=_CAMPAIGN_SOURCE)

    out_dir = pytester.path / "out"
    result = pytester.runpytest_subprocess("--runs", "2", "--out", str(out_dir))

    result.assert_outcomes(passed=2)
    written = sorted(out_dir.glob("run_*.h5"))
    assert len(written) == 2

    cases = [h5py.File(path, "r") for path in written]
    try:
        gyro_0 = np.array(cases[0]["gyro"])
        gyro_1 = np.array(cases[1]["gyro"])
        assert not np.allclose(gyro_0, gyro_1), "both cases recorded identical IMU noise"

        for case in cases:
            for name in ("sigma0_BN", "omega0_BN_B", "gyro_bias", "accel_bias"):
                assert f"draw_{name}" in case.attrs, f"draw_{name} missing from {case.filename}"
    finally:
        for case in cases:
            case.close()


def test_default_out_dir_is_shared_across_xdist_workers_without_out(pytester, monkeypatch):
    """Without `--out`, `pytest_configure` falls back to `tempfile.mkdtemp`; under `-n W` each
    worker is its own process, so without `pytest_configure_node` handing every worker the
    controller's own resolved directory, each would pick its own and a campaign's files would
    scatter. `TMPDIR` is redirected into this test's own tmp path only so the test can find
    where the default directory landed; it plays no part in the plugin's own logic."""
    monkeypatch.setenv("TMPDIR", str(pytester.path))
    pytester.makepyfile(test_campaign=_CAMPAIGN_SOURCE)

    result = pytester.runpytest_subprocess("-n", "2", "--runs", "2")

    result.assert_outcomes(passed=2)
    run_files = list(pytester.path.rglob("run_*.h5"))
    assert len(run_files) == 2
    out_dirs = {path.parent for path in run_files}
    assert len(out_dirs) == 1, f"run files scattered across directories: {out_dirs}"


def test_campaign_runs_with_xdist_absent(pytester):
    """`-p no:xdist` leaves xdist's own hookspecs unregistered -- the same state as xdist not
    being installed at all. Loaded via the `pytest11` entry point, this plugin configures that
    session regardless, so its `pytest_configure_node` (an xdist hookimpl with no matching
    hookspec there) must not make pytest itself refuse to start."""
    pytester.makepyfile(test_campaign=_CAMPAIGN_SOURCE)

    out_dir = pytester.path / "out"
    result = pytester.runpytest_subprocess("-p", "no:xdist", "--runs", "1", "--out", str(out_dir))

    result.assert_outcomes(passed=1)
    assert len(list(out_dir.glob("run_*.h5"))) == 1


def test_a_session_with_no_case_creates_no_out_dir(pytester, monkeypatch):
    """`pytest_configure` computes the default `out_dir` path without creating it, because this
    plugin, loaded via the `pytest11` entry point, configures every session of a consumer that
    has `wise-dynamics` installed -- most of which run no case and call `run_writer` never.
    `TMPDIR` is redirected into this test's own tmp path only so the test can assert nothing
    landed there; it plays no part in the plugin's own logic."""
    monkeypatch.setenv("TMPDIR", str(pytester.path))
    pytester.makepyfile(test_nothing="def test_nothing():\n    pass\n")

    result = pytester.runpytest_subprocess()

    result.assert_outcomes(passed=1)
    assert not list(pytester.path.glob("wise_dynamics_*"))
