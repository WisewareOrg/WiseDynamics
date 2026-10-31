"""Smoke tests, run before anything else per SPEC-demo.md §3: a single-process 300 s run, then
the untested Python-module-in-a-Monte-Carlo-worker pattern over 4 runs. If the Monte Carlo step
fails for pickling or import reasons, that is reported rather than worked around."""

import numpy as np
from gnc_algorithms.mrp import angle, subtract

from sim.basilisk.campaigns.imu_dead_reckoning_mc import build_controller
from sim.basilisk.scenarios.imu_dead_reckoning import run


def test_attitude_estimate_tracks_truth_at_ten_seconds():
    result = run(duration_s=300.0)
    idx = np.argmin(np.abs(result.time_s - 10.0))

    sigma_err = subtract(result.truth_sigma_BN[idx], result.est_sigma_BN[idx])
    error_deg = np.degrees(angle(sigma_err))

    assert error_deg < 1.0, f"attitude error at t=10s was {error_deg} deg, expected < 1 deg"


def test_position_error_small_and_grows_with_time():
    result = run(duration_s=300.0)
    position_error = np.linalg.norm(result.truth_r_BN_N - result.est_r_BN_N, axis=1)

    assert position_error[-1] < 100.0, (
        f"position error at t=300s was {position_error[-1]} m, expected < 100 m"
    )

    slope, _ = np.polyfit(result.time_s, position_error, 1)
    assert slope > 0.0, f"position error should grow with time, fitted slope was {slope}"


def test_four_run_monte_carlo_through_controller_has_zero_failures(tmp_path):
    """The untested Python-module-in-a-Monte-Carlo-worker pattern: the Controller's
    `multiprocessing.Pool` workers must be able to import this scenario module by its real
    dotted name to pickle the creation/execution functions across the process boundary."""
    controller = build_controller(
        num_runs=4, archive_dir=str(tmp_path / "mc_smoke"), worker_count=4
    )
    failures = controller.executeSimulations()

    assert failures == [], f"Monte Carlo smoke run had failures: {failures}"
