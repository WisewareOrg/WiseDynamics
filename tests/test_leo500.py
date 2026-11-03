"""`Leo500`: its nominal environment and placement, how its keywords reach the Basilisk objects,
and what it refuses."""

import inspect
import math

import numpy as np
import pytest
from Basilisk.utilities import orbitalMotion

from wise_dynamics.basilisk import ImuSpacecraft, Leo500

D2R = math.pi / 180.0

PLACEMENT = {
    "semi_major_axis_m": 7000.0e3,
    "eccentricity": 0.01,
    "inclination_rad": 30.0 * D2R,
    "raan_rad": 10.0 * D2R,
    "arg_periapsis_rad": 20.0 * D2R,
    "true_anomaly_rad": 40.0 * D2R,
    "sigma0_BN": (0.05, -0.1, 0.2),
    "omega0_BN_B_rad_s": (0.01, -0.02, 0.03),
    "step_s": 0.5,
}
"""Constructor keywords differing from nominal in every parameter."""


def test_nominal_scenario_is_the_specified_one():
    """Read from the class, as a consumer reads it without building a scenario."""
    assert Leo500.mu_m3_s2 == {"earth": 3.986004360e14}
    assert Leo500.semi_major_axis_m == 6871.0e3
    assert Leo500.eccentricity == 0.0001
    assert Leo500.inclination_rad == 45.0 * D2R
    assert Leo500.raan_rad == 48.2 * D2R
    assert Leo500.arg_periapsis_rad == 347.8 * D2R
    assert Leo500.true_anomaly_rad == 85.3 * D2R
    assert Leo500.sigma0_BN == (0.1, 0.2, -0.3)
    assert Leo500.omega0_BN_B_rad_s == (D2R, D2R, D2R)
    assert Leo500.step_s == 1.0
    assert Leo500.standard_dispersion == {"sigma0_BN": 0.01, "omega0_BN_B_rad_s": 0.2 * D2R}


def test_each_keyword_defaults_to_its_class_attribute():
    """So a scenario built with no keywords is the scenario a consumer reads from the class."""
    parameters = dict(inspect.signature(Leo500).parameters)
    assert parameters.pop("vehicle").default is inspect.Parameter.empty
    assert len(parameters) == 9
    for name, parameter in parameters.items():
        assert parameter.default == getattr(Leo500, name), name


def test_keywords_reach_the_basilisk_objects():
    vehicle = ImuSpacecraft()
    scenario = Leo500(vehicle, **PLACEMENT)
    (earth,) = vehicle.spacecraft.gravField.gravBodies
    assert earth.isCentralBody
    assert earth.mu == Leo500.mu_m3_s2["earth"]
    assert scenario._simulation.TaskList[0].TaskData.TaskPeriod == 500_000_000
    assert scenario._simulation.TaskList[0].TaskModels == vehicle.models()

    state = scenario.initial_state()
    assert np.array_equal(state["sigma0_BN"], PLACEMENT["sigma0_BN"])
    assert np.array_equal(state["omega0_BN_B"], PLACEMENT["omega0_BN_B_rad_s"])
    elements = orbitalMotion.rv2elem(Leo500.mu_m3_s2["earth"], state["r0_N"], state["v0_N"])
    assert elements.a == pytest.approx(PLACEMENT["semi_major_axis_m"], rel=1e-12)
    assert elements.e == pytest.approx(PLACEMENT["eccentricity"], rel=1e-9)
    assert elements.i == pytest.approx(PLACEMENT["inclination_rad"], abs=1e-12)
    assert elements.Omega == pytest.approx(PLACEMENT["raan_rad"], abs=1e-12)
    assert elements.omega == pytest.approx(PLACEMENT["arg_periapsis_rad"], abs=1e-9)
    assert elements.f == pytest.approx(PLACEMENT["true_anomaly_rad"], abs=1e-9)


def test_draw_adds_normal_samples_to_the_defaults_in_dispersion_order():
    draws = Leo500.draw(np.random.default_rng(7))
    rng = np.random.default_rng(7)
    assert list(draws) == list(Leo500.standard_dispersion)
    for name, std in Leo500.standard_dispersion.items():
        expected = np.asarray(getattr(Leo500, name)) + rng.normal(0.0, std, size=3)
        assert np.array_equal(draws[name], expected), name


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"semi_major_axis_m": 0.0}, "semi-major axis must be positive"),
        ({"eccentricity": 1.0}, r"eccentricity must be in \[0, 1\)"),
        ({"eccentricity": -0.1}, r"eccentricity must be in \[0, 1\)"),
        ({"step_s": 0.0}, "step must be positive"),
    ],
)
def test_a_refused_placement_leaves_the_vehicle_unplaced(keywords, message):
    vehicle = ImuSpacecraft()
    with pytest.raises(ValueError, match=message):
        Leo500(vehicle, **keywords)
    assert len(vehicle.spacecraft.gravField.gravBodies) == 0
