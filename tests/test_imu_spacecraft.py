"""`ImuSpacecraft`: its nominal vehicle, how a configuration reaches Basilisk, and the IMU's
measured behaviour."""

import dataclasses
import math

import numpy as np
import pytest
from Basilisk.utilities import orbitalMotion

from tests.support import run_with_stub
from wise_dynamics.basilisk import ImuSpacecraft, ImuSpacecraftConfig, OrbitElements, RigidHub

D2R = math.pi / 180.0

QUIET = ImuSpacecraftConfig(
    omega0_BN_B=(0.0, 0.0, 0.0),
    gyro_bias_P=(0.0, 0.0, 0.0),
    accel_bias_P=(0.0, 0.0, 0.0),
)
"""Zero rate and zero bias: the IMU then reports its white noise alone."""

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

    truth = run_with_stub(plant, duration_s=10.0)["truth"]
    assert np.allclose(np.array(truth.sigma_BN)[0], state["sigma0_BN"], rtol=0.0, atol=1e-15)
    assert np.allclose(np.array(truth.omega_BN_B)[0], state["omega0_BN_B"], rtol=0.0, atol=1e-15)
    assert np.allclose(np.array(truth.r_BN_N)[0], state["r0_N"], rtol=0.0, atol=1e-6)
    assert np.allclose(np.array(truth.v_BN_N)[0], state["v0_N"], rtol=0.0, atol=1e-9)


def test_noiseless_accelerometer_reads_zero_in_orbit_and_first_samples_read_zero():
    """The accelerometer measures non-gravitational specific force, so with zero bias and noise
    it reads zero in free fall; Basilisk writes the IMU's first output after the first step,
    so the t = 0 sample is zero on both instruments."""
    quiet = dataclasses.replace(QUIET, gyro_noise_std_rad_s=0.0, accel_noise_std_m_s2=0.0)
    imu = run_with_stub(ImuSpacecraft(quiet), duration_s=300.0)["imu"]
    accel = np.array(imu.AccelPlatform)
    gyro = np.array(imu.AngVelPlatform)
    assert np.array_equal(accel[0], np.zeros(3))
    assert np.array_equal(gyro[0], np.zeros(3))
    assert np.max(np.abs(accel)) < 1e-12


def test_recorded_imu_mean_is_the_configured_bias():
    config = dataclasses.replace(
        QUIET, gyro_bias_P=DISPERSED.gyro_bias_P, accel_bias_P=DISPERSED.accel_bias_P
    )
    imu = run_with_stub(ImuSpacecraft(config, noise_seed=3), duration_s=500.0, record_s=1.0)["imu"]
    # Row 0 is the t = 0 sample, written before the IMU's first output.
    gyro_mean = np.array(imu.AngVelPlatform)[1:].mean(axis=0)
    accel_mean = np.array(imu.AccelPlatform)[1:].mean(axis=0)
    assert np.allclose(gyro_mean, config.gyro_bias_P, rtol=0.0, atol=1e-5)
    assert np.allclose(accel_mean, config.accel_bias_P, rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("step_s", [1.0, 0.1])
def test_noise_std_is_per_sample_at_any_step(step_s):
    """`gyro_noise_std_rad_s` and `accel_noise_std_m_s2` are per-sample standard deviations, so
    the recorded spread is the configured value whatever the step."""
    sigma = 5e-5  # the settled per-sample 1-sigma, stated here independently of the plant
    config = dataclasses.replace(QUIET, step_s=step_s)
    imu = run_with_stub(ImuSpacecraft(config), duration_s=500.0 * step_s, record_s=step_s)["imu"]
    # Row 0 is the t = 0 sample, written before the IMU's first output.
    for samples in (np.array(imu.AngVelPlatform)[1:], np.array(imu.AccelPlatform)[1:]):
        for axis_std in samples.std(axis=0):
            assert abs(axis_std - sigma) / sigma < 0.15, f"std {axis_std} vs {sigma}"


def _gyro_noise(noise_seed: int) -> np.ndarray:
    imu = run_with_stub(ImuSpacecraft(QUIET, noise_seed=noise_seed), 50.0, record_s=1.0)["imu"]
    return np.array(imu.AngVelPlatform)


def test_noise_seed_reproduces_and_distinguishes_noise():
    assert np.array_equal(_gyro_noise(7), _gyro_noise(7))
    assert not np.allclose(_gyro_noise(0), _gyro_noise(1))


def test_imu_reader_is_subscribed_to_the_imu_output():
    plant = ImuSpacecraft()
    reader = plant.readers()["imu"]
    assert reader.isLinked()
    imu = run_with_stub(plant, duration_s=20.0)["imu"]
    assert np.array_equal(reader().AngVelPlatform, np.array(imu.AngVelPlatform)[-1])


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
