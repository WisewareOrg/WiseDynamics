"""`Leo400`: `Leo500`'s vehicle, environment and dispersions on a 400 km orbit."""

import numpy as np
import pytest

from wise_dynamics.basilisk import SCENARIOS
from wise_dynamics.basilisk.scenarios.leo400 import Leo400
from wise_dynamics.basilisk.scenarios.leo500 import Leo500

SHARED = (
    "mu_m3_s2",
    "eccentricity",
    "inclination_rad",
    "raan_rad",
    "arg_periapsis_rad",
    "true_anomaly_rad",
    "sigma0_BN",
    "omega0_BN_B_rad_s",
    "record_s",
)


def test_leo400_is_registered_under_its_name():
    """The `sim_scenario` mark reaches it as `"leo400"`."""
    assert SCENARIOS["leo400"] is Leo400


def test_leo400_differs_from_leo500_only_in_its_semi_major_axis():
    """6771 km, about 393 km above the 6378 km equatorial radius; every other attribute shared."""
    assert Leo400.semi_major_axis_m == 6771.0e3
    for name in SHARED:
        assert getattr(Leo400, name) == getattr(Leo500, name), name


def test_leo400_places_its_vehicle_on_its_own_orbit():
    """The initial position's radius is within the eccentricity of 6771 km, not of 6871 km."""
    radius = float(np.linalg.norm(Leo400().initial_state()["r0_N"]))

    assert radius == pytest.approx(6771.0e3, rel=Leo400.eccentricity)
