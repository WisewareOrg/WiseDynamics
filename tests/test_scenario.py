"""Smoke tests, run before anything else per SPEC-demo.md §3: a single-process 300 s run of the
nominal case. The pytest-based campaign (dispersed cases, run export) is exercised by running
`sim/basilisk/scenarios` itself, not from here."""

import numpy as np
from gnc_algorithms.mrp import angle, subtract

from sim.basilisk.scenarios.test_imu_dead_reckoning import run


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
