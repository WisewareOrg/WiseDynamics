"""`Leo500`: its prebuilt environment and placement, its Basilisk wiring, and its dispersion."""

import inspect
import math

import numpy as np
import pytest
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.utilities import orbitalMotion

from wise_dynamics.basilisk import ImuSpacecraft, Leo500

D2R = math.pi / 180.0


class _Guidance(inertial3D.inertial3D):
    """Basilisk's stock `inertial3D` standing in for flight software, stepping at 1 Hz.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 1.0


class Placed(Leo500):
    """A scenario differing from `Leo500` in every placement attribute.

    Attributes:
        semi_major_axis_m: Semi-major axis of the initial orbit, in metres.
        eccentricity: Eccentricity of the initial orbit.
        inclination_rad: Inclination, in radians.
        raan_rad: Right ascension of the ascending node, in radians.
        arg_periapsis_rad: Argument of periapsis, in radians.
        true_anomaly_rad: True anomaly at t = 0, in radians.
        sigma0_BN: Initial attitude, the MRP of the body frame relative to N.
        omega0_BN_B_rad_s: Initial body rate relative to N, in body axes, in rad/s.
    """

    semi_major_axis_m = 7000.0e3
    eccentricity = 0.01
    inclination_rad = 30.0 * D2R
    raan_rad = 10.0 * D2R
    arg_periapsis_rad = 20.0 * D2R
    true_anomaly_rad = 40.0 * D2R
    sigma0_BN = (0.05, -0.1, 0.2)
    omega0_BN_B_rad_s = (0.01, -0.02, 0.03)


def test_the_scenario_is_its_class_attributes():
    """Read from the class, as a consumer reads it without building a scenario."""
    assert inspect.signature(Leo500).parameters == {}
    assert Leo500.fidelity == "lo"
    assert Leo500.mu_m3_s2 == {"earth": 3.986004360e14}
    assert Leo500.semi_major_axis_m == 6871.0e3
    assert Leo500.eccentricity == 0.0001
    assert Leo500.inclination_rad == 45.0 * D2R
    assert Leo500.raan_rad == 48.2 * D2R
    assert Leo500.arg_periapsis_rad == 347.8 * D2R
    assert Leo500.true_anomaly_rad == 85.3 * D2R
    assert Leo500.sigma0_BN == (0.1, 0.2, -0.3)
    assert Leo500.omega0_BN_B_rad_s == (D2R, D2R, D2R)
    assert Leo500.record_s == 10.0


def test_the_placement_reaches_the_basilisk_objects():
    """Gravity and every placement attribute reach the hub and the task the run builds."""
    scenario = Placed()
    vehicle = scenario.vehicle
    state = scenario.initial_state()
    scenario.run(_Guidance(), {}, duration_s=1.0)

    (earth,) = vehicle.spacecraft.gravField.gravBodies
    assert earth.isCentralBody
    assert earth.mu == Leo500.mu_m3_s2["earth"]
    dynamics, _, _ = scenario._simulation.TaskList
    assert dynamics.TaskModels[:2] == vehicle.models()
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


def test_the_scenario_builds_and_holds_its_vehicle():
    """The constructor builds an `ImuSpacecraft`, holds it in `vehicles`, and has drawn nothing."""
    scenario = Leo500()
    assert type(scenario.vehicle) is ImuSpacecraft
    assert scenario.vehicles == [scenario.vehicle]
    assert scenario.draws == {}


def test_disperse_draws_the_placement_then_the_vehicle_and_builds_the_vehicle_from_them():
    """`disperse` draws the placement then the vehicle's biases, and builds with them."""
    scenario = Leo500()
    built = scenario.vehicle
    scenario.disperse(np.random.default_rng(7), noise_seed=11)

    rng = np.random.default_rng(7)
    expected = {}
    for name, nominal, std in (
        ("sigma0_BN", Leo500.sigma0_BN, 0.01),
        ("omega0_BN_B_rad_s", Leo500.omega0_BN_B_rad_s, 0.2 * D2R),
        ("gyro_bias_P_rad_s", ImuSpacecraft.gyro_bias_P_rad_s, 1e-4),
        ("accel_bias_P_m_s2", ImuSpacecraft.accel_bias_P_m_s2, 1e-4),
    ):
        expected[name] = np.asarray(nominal) + rng.normal(0.0, std, size=3)
    assert list(scenario.draws) == list(expected)
    for name, value in expected.items():
        assert np.array_equal(scenario.draws[name], value), name

    assert Leo500.sigma0_BN == (0.1, 0.2, -0.3)
    assert scenario.vehicle is not built
    assert scenario.vehicles == [scenario.vehicle]
    hub = scenario.vehicle.spacecraft.hub
    assert np.array_equal(np.array(hub.sigma_BNInit).flatten(), expected["sigma0_BN"])
    assert np.array_equal(np.array(hub.omega_BN_BInit).flatten(), expected["omega0_BN_B_rad_s"])
    assert len(scenario.vehicle.spacecraft.gravField.gravBodies) == 1
    imu = scenario.vehicle.imu
    assert np.array_equal(np.array(imu.senRotBias).flatten(), expected["gyro_bias_P_rad_s"])
    assert np.array_equal(np.array(imu.senTransBias).flatten(), expected["accel_bias_P_m_s2"])
    assert imu.RNGSeed == 11
