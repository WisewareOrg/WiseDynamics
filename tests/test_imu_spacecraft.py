"""`ImuSpacecraft`: its nominal vehicle, how its configuration reaches the Basilisk objects it
builds, its validation, and its reader's lifetime."""

import gc
import math
import weakref

import numpy as np
import pytest
from Basilisk.utilities import orbitalMotion

from tests.support import run_with_stub
from wise_dynamics.basilisk import ImuSpacecraft, ImuSpacecraftConfig, OrbitElements, RigidHub

D2R = math.pi / 180.0


DISPERSED = ImuSpacecraftConfig(
    orbit=OrbitElements(
        semi_major_axis_m=7000.0e3,
        eccentricity=0.01,
        inclination_rad=30.0 * D2R,
        raan_rad=10.0 * D2R,
        arg_periapsis_rad=20.0 * D2R,
        true_anomaly_rad=40.0 * D2R,
    ),
    hub=RigidHub(
        mass_kg=500.0, inertia_B_kg_m2=((150.0, -1.0, 0.0), (-1.0, 200.0, 0.0), (0.0, 0.0, 250.0))
    ),
    sigma0_BN=(0.05, -0.1, 0.2),
    omega0_BN_B=(0.01, -0.02, 0.03),
    gyro_bias_P=(3e-4, -1e-4, 2e-4),
    accel_bias_P=(-2e-4, 4e-4, 1e-4),
    gyro_noise_std_rad_s=7e-5,
    accel_noise_std_m_s2=9e-5,
    step_s=0.5,
)
"""A configuration differing from nominal in every field."""


def test_nominal_vehicle_is_the_specified_one():
    nominal = ImuSpacecraft.nominal()
    assert nominal == ImuSpacecraftConfig()
    assert nominal.orbit == OrbitElements(
        semi_major_axis_m=6871.0e3,
        eccentricity=0.0001,
        inclination_rad=45.0 * D2R,
        raan_rad=48.2 * D2R,
        arg_periapsis_rad=347.8 * D2R,
        true_anomaly_rad=85.3 * D2R,
    )
    assert nominal.hub == RigidHub(
        mass_kg=750.0, inertia_B_kg_m2=((900.0, 0.0, 0.0), (0.0, 800.0, 0.0), (0.0, 0.0, 600.0))
    )
    assert nominal.sigma0_BN == (0.1, 0.2, -0.3)
    assert nominal.omega0_BN_B == (D2R, D2R, D2R)
    assert nominal.gyro_bias_P == (1e-4, -2e-4, 5e-5)
    assert nominal.accel_bias_P == (1e-4, -5e-5, 2e-4)
    assert nominal.gyro_noise_std_rad_s == 5e-5
    assert nominal.accel_noise_std_m_s2 == 5e-5
    assert nominal.step_s == 1.0


def test_configuration_reaches_the_basilisk_objects():
    plant = ImuSpacecraft(DISPERSED, noise_seed=11)
    hub = plant.spacecraft.hub
    assert hub.mHub == 500.0
    assert np.array_equal(np.array(hub.IHubPntBc_B), np.array(DISPERSED.hub.inertia_B_kg_m2))

    imu = plant.imu
    assert np.array_equal(np.array(imu.senRotBias).flatten(), DISPERSED.gyro_bias_P)
    assert np.array_equal(np.array(imu.senTransBias).flatten(), DISPERSED.accel_bias_P)
    assert np.array_equal(np.array(imu.PMatrixGyro), 7e-5 * np.eye(3))
    assert np.array_equal(np.array(imu.PMatrixAccel), 9e-5 * np.eye(3))
    assert np.array_equal(np.array(imu.dcm_PB), np.eye(3))
    assert np.array_equal(np.array(imu.sensorPos_B).flatten(), np.zeros(3))
    (earth,) = plant.spacecraft.gravField.gravBodies
    assert earth.isCentralBody
    assert imu.RNGSeed == 11

    state = plant.initial_state()
    elements = orbitalMotion.rv2elem(plant.mu, state["r0_N"], state["v0_N"])
    orbit = DISPERSED.orbit
    assert elements.a == pytest.approx(orbit.semi_major_axis_m, rel=1e-12)
    assert elements.e == pytest.approx(orbit.eccentricity, rel=1e-9)
    assert elements.i == pytest.approx(orbit.inclination_rad, abs=1e-12)
    assert elements.Omega == pytest.approx(orbit.raan_rad, abs=1e-12)
    assert elements.omega == pytest.approx(orbit.arg_periapsis_rad, abs=1e-9)
    assert elements.f == pytest.approx(orbit.true_anomaly_rad, abs=1e-9)


def test_initial_state_is_the_configuration_and_the_truth_at_t0():
    plant = ImuSpacecraft(DISPERSED)
    state = plant.initial_state()
    r_N, v_N = DISPERSED.orbit.to_inertial(plant.mu)
    assert np.array_equal(state["sigma0_BN"], DISPERSED.sigma0_BN)
    assert np.array_equal(state["omega0_BN_B"], DISPERSED.omega0_BN_B)
    assert np.array_equal(state["r0_N"], r_N)
    assert np.array_equal(state["v0_N"], v_N)

    truth = run_with_stub(plant, duration_s=10.0).datasets(plant.recorded_fields)
    for dataset, key, tolerance in (
        ("truth/sigma_BN", "sigma0_BN", 1e-15),
        ("truth/omega_BN_B", "omega0_BN_B", 1e-15),
        ("truth/r_N", "r0_N", 1e-6),
        ("truth/v_N", "v0_N", 1e-9),
    ):
        assert np.allclose(truth[dataset][0][0], state[key], rtol=0.0, atol=tolerance), key


def test_imu_reader_is_subscribed_to_the_imu_output():
    plant = ImuSpacecraft()
    reader = plant.readers()["imu"]
    assert reader.isLinked()
    gyro, _ = run_with_stub(plant, duration_s=20.0).datasets(plant.recorded_fields)["imu/gyro_P"]
    assert np.array_equal(reader().AngVelPlatform, gyro[-1])


def test_a_plant_runs_once():
    plant = ImuSpacecraft()
    plant.execute(1.0)
    with pytest.raises(RuntimeError, match="already run"):
        plant.execute(1.0)


@pytest.mark.parametrize(
    ("elements", "message"),
    [
        ({"semi_major_axis_m": 0.0}, "semi-major axis must be positive"),
        ({"eccentricity": 1.0}, r"eccentricity must be in \[0, 1\)"),
        ({"eccentricity": -0.1}, r"eccentricity must be in \[0, 1\)"),
    ],
)
def test_orbit_elements_outside_a_closed_orbit_are_refused(elements, message):
    with pytest.raises(ValueError, match=message):
        OrbitElements(**elements)


def test_a_plant_with_a_non_positive_step_is_refused():
    with pytest.raises(ValueError, match="step must be positive"):
        ImuSpacecraft(ImuSpacecraftConfig(step_s=0.0))


def test_a_reader_reads_the_live_value_after_its_plant_is_released():
    """A reader holds its plant, so reading it after the consumer releases its own reference to
    the plant gives the plant's output, not freed memory."""
    plant = ImuSpacecraft()
    released = weakref.ref(plant)
    reader = plant.readers()["imu"]
    run_with_stub(plant, duration_s=20.0)
    gyro = list(reader().AngVelPlatform)
    del plant
    gc.collect()
    others = [ImuSpacecraft() for _ in range(10)]

    assert others
    assert released() is not None, "the reader released its plant"
    assert any(component != 0.0 for component in gyro)
    assert list(reader().AngVelPlatform) == gyro
