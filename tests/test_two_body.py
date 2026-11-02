"""The nominal plant's translational motion over one orbit against analytic two-body motion."""

import math

import numpy as np

from tests.support import EARTH_MU_M3_S2, run_with_stub
from wise_dynamics.basilisk import ImuSpacecraft, OrbitElements

DURATION_S = 5700.0

# Measured after one orbit with bsk 2.12.1: 6e-7 m and 7e-10 m/s from the analytic motion.
POSITION_TOL_M = 1e-4
VELOCITY_TOL_M_S = 1e-7


def kepler_state(orbit: OrbitElements, mu: float, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Inertial position and velocity at `t_s` of a two-body orbit starting at `orbit`."""
    a, e = orbit.semi_major_axis_m, orbit.eccentricity
    eccentric0 = 2.0 * math.atan(
        math.sqrt((1 - e) / (1 + e)) * math.tan(orbit.true_anomaly_rad / 2)
    )
    mean = eccentric0 - e * math.sin(eccentric0) + math.sqrt(mu / a**3) * t_s
    eccentric = mean
    for _ in range(20):
        eccentric -= (eccentric - e * math.sin(eccentric) - mean) / (1 - e * math.cos(eccentric))
    true = 2.0 * math.atan2(
        math.sqrt(1 + e) * math.sin(eccentric / 2), math.sqrt(1 - e) * math.cos(eccentric / 2)
    )

    p = a * (1 - e**2)
    radius = p / (1 + e * math.cos(true))
    r_perifocal = radius * np.array([math.cos(true), math.sin(true), 0.0])
    v_perifocal = math.sqrt(mu / p) * np.array([-math.sin(true), e + math.cos(true), 0.0])

    cO, sO = math.cos(orbit.raan_rad), math.sin(orbit.raan_rad)
    ci, si = math.cos(orbit.inclination_rad), math.sin(orbit.inclination_rad)
    cw, sw = math.cos(orbit.arg_periapsis_rad), math.sin(orbit.arg_periapsis_rad)
    perifocal_to_N = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ]
    )
    return perifocal_to_N @ r_perifocal, perifocal_to_N @ v_perifocal


def test_one_orbit_follows_two_body_motion():
    plant = ImuSpacecraft()
    truth = run_with_stub(plant, duration_s=DURATION_S, record_s=100.0)["truth"]
    r_N, v_N = kepler_state(ImuSpacecraft.nominal().orbit, EARTH_MU_M3_S2, DURATION_S)

    position_error_m = np.linalg.norm(np.array(truth.r_BN_N)[-1] - r_N)
    velocity_error_m_s = np.linalg.norm(np.array(truth.v_BN_N)[-1] - v_N)
    assert position_error_m < POSITION_TOL_M, f"position off by {position_error_m} m"
    assert velocity_error_m_s < VELOCITY_TOL_M_S, f"velocity off by {velocity_error_m_s} m/s"


def test_the_orbit_follows_the_class_mu():
    """A plant whose class `mu` differs from Basilisk's Earth default moves on the two-body orbit
    of its own `mu`."""

    class OtherMu(ImuSpacecraft):
        mu = EARTH_MU_M3_S2 * 1.001

    duration_s = 1000.0
    truth = run_with_stub(OtherMu(), duration_s=duration_s, record_s=100.0)["truth"]
    r_N, v_N = kepler_state(OtherMu.nominal().orbit, OtherMu.mu, duration_s)
    assert np.linalg.norm(np.array(truth.r_BN_N)[-1] - r_N) < POSITION_TOL_M
    assert np.linalg.norm(np.array(truth.v_BN_N)[-1] - v_N) < VELOCITY_TOL_M_S
