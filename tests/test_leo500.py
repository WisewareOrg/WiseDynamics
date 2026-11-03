"""`Leo500`: its prebuilt environment and placement, its draws and dispersion, how its placement
reaches the Basilisk objects, and what it refuses."""

import inspect
import math

import numpy as np
import pytest
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.utilities import orbitalMotion

from wise_dynamics.basilisk import ImuSpacecraft, Leo500

D2R = math.pi / 180.0


class Placed(Leo500):
    """A scenario differing from `Leo500` in every placement attribute."""

    semi_major_axis_m = 7000.0e3
    eccentricity = 0.01
    inclination_rad = 30.0 * D2R
    raan_rad = 10.0 * D2R
    arg_periapsis_rad = 20.0 * D2R
    true_anomaly_rad = 40.0 * D2R
    sigma0_BN = (0.05, -0.1, 0.2)
    omega0_BN_B_rad_s = (0.01, -0.02, 0.03)
    step_s = 0.5


def test_the_scenario_is_its_class_attributes():
    """Read from the class, as a consumer reads it without building a scenario."""
    assert inspect.signature(Leo500).parameters == {}
    assert Leo500.mu_m3_s2 == {"earth": 3.986004360e14}
    assert Leo500.semi_major_axis_m == 6871.0e3
    assert Leo500.eccentricity == 0.0001
    assert Leo500.inclination_rad == 45.0 * D2R
    assert Leo500.raan_rad == 48.2 * D2R
    assert Leo500.arg_periapsis_rad == 347.8 * D2R
    assert Leo500.true_anomaly_rad == 85.3 * D2R
    assert Leo500.sigma0_BN == (0.1, 0.2, -0.3)
    assert Leo500.omega0_BN_B_rad_s == (D2R, D2R, D2R)
    assert Leo500.step_s == 0.02
    assert Leo500.standard_dispersion == {"sigma0_BN": 0.01, "omega0_BN_B_rad_s": 0.2 * D2R}


def test_the_placement_reaches_the_basilisk_objects():
    vehicle = ImuSpacecraft()
    scenario = Placed()
    scenario.add(vehicle)
    state = scenario.initial_state()
    scenario.run(inertial3D.inertial3D(), fsw_fields={}, duration_s=1.0)

    (earth,) = vehicle.spacecraft.gravField.gravBodies
    assert earth.isCentralBody
    assert earth.mu == Leo500.mu_m3_s2["earth"]
    (task,) = scenario._simulation.TaskList
    assert task.TaskData.TaskPeriod == 500_000_000
    assert task.TaskModels[:2] == vehicle.models()
    hub = vehicle.spacecraft.hub
    assert np.array_equal(np.array(hub.sigma_BNInit).flatten(), Placed.sigma0_BN)
    assert np.array_equal(np.array(hub.omega_BN_BInit).flatten(), Placed.omega0_BN_B_rad_s)
    assert np.array_equal(np.array(hub.r_CN_NInit).flatten(), state["r0_N"])
    assert np.array_equal(np.array(hub.v_CN_NInit).flatten(), state["v0_N"])

    elements = orbitalMotion.rv2elem(Leo500.mu_m3_s2["earth"], state["r0_N"], state["v0_N"])
    assert elements.a == pytest.approx(Placed.semi_major_axis_m, rel=1e-12)
    assert elements.e == pytest.approx(Placed.eccentricity, rel=1e-9)
    assert elements.i == pytest.approx(Placed.inclination_rad, abs=1e-12)
    assert elements.Omega == pytest.approx(Placed.raan_rad, abs=1e-12)
    assert elements.omega == pytest.approx(Placed.arg_periapsis_rad, abs=1e-9)
    assert elements.f == pytest.approx(Placed.true_anomaly_rad, abs=1e-9)


def test_draw_adds_normal_samples_to_the_defaults_in_dispersion_order():
    draws = Leo500.draw(np.random.default_rng(7))
    rng = np.random.default_rng(7)
    assert list(draws) == list(Leo500.standard_dispersion)
    for name, std in Leo500.standard_dispersion.items():
        expected = np.asarray(getattr(Leo500, name)) + rng.normal(0.0, std, size=3)
        assert np.array_equal(draws[name], expected), name


def test_disperse_overrides_the_placement_of_one_instance():
    scenario = Leo500()
    scenario.disperse({"sigma0_BN": (0.3, -0.2, 0.1)})
    assert np.array_equal(scenario.initial_state()["sigma0_BN"], [0.3, -0.2, 0.1])
    assert Leo500.sigma0_BN == (0.1, 0.2, -0.3)
    assert np.array_equal(Leo500().initial_state()["sigma0_BN"], Leo500.sigma0_BN)


def test_disperse_refuses_a_placement_outside_the_standard_dispersion():
    scenario = Leo500()
    with pytest.raises(ValueError, match=r"Leo500 does not disperse \['semi_major_axis_m'\]"):
        scenario.disperse({"sigma0_BN": (0.3, -0.2, 0.1), "semi_major_axis_m": 7000.0e3})
    assert scenario.sigma0_BN == Leo500.sigma0_BN


def test_a_vehicle_added_after_the_run_is_refused():
    scenario = Leo500()
    scenario.add(ImuSpacecraft())
    scenario.run(inertial3D.inertial3D(), fsw_fields={}, duration_s=1.0)
    with pytest.raises(RuntimeError, match="already run"):
        scenario.add(ImuSpacecraft())


def test_a_second_vehicle_is_refused():
    scenario = Leo500()
    first = ImuSpacecraft()
    scenario.add(first)
    with pytest.raises(ValueError, match="a second one needs a placement of its own"):
        scenario.add(ImuSpacecraft())
    assert scenario.vehicles == [first]


class Hyperbolic(Leo500):
    eccentricity = 1.0


class Collapsed(Leo500):
    semi_major_axis_m = 0.0


class Stalled(Leo500):
    step_s = 0.0


@pytest.mark.parametrize(
    ("scenario_type", "message"),
    [
        (Collapsed, "semi-major axis must be positive"),
        (Hyperbolic, r"eccentricity must be in \[0, 1\)"),
        (Stalled, "step must be positive"),
    ],
)
def test_a_refused_placement_leaves_the_vehicle_unplaced(scenario_type, message):
    vehicle = ImuSpacecraft()
    scenario = scenario_type()
    scenario.add(vehicle)
    with pytest.raises(ValueError, match=message):
        scenario.run(inertial3D.inertial3D(), fsw_fields={}, duration_s=1.0)
    assert len(vehicle.spacecraft.gravField.gravBodies) == 0
    assert len(scenario._simulation.TaskList) == 0
