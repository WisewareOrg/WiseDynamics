"""`ImuSpacecraft`: its nominal vehicle, its wiring into Basilisk, and what it records.

The tests cover how its keywords reach the Basilisk objects it builds, its dispersion, its reader,
and its recorded fields.
"""

import inspect

import numpy as np
import pytest
from Basilisk.fswAlgorithms import inertial3D

from wise_dynamics.basilisk import ImuSpacecraft
from wise_dynamics.basilisk.scenarios.leo500 import Leo500


class _Guidance(inertial3D.inertial3D):
    """Basilisk's stock `inertial3D` standing in for flight software, stepping at 1 Hz.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 1.0


DISPERSED = {
    "mass_kg": 500.0,
    "inertia_B_kg_m2": ((150.0, -1.0, 0.0), (-1.0, 200.0, 0.0), (0.0, 0.0, 250.0)),
    "gyro_bias_P_rad_s": (3e-4, -1e-4, 2e-4),
    "accel_bias_P_m_s2": (-2e-4, 4e-4, 1e-4),
    "gyro_noise_density_rad_s_rthz": 7e-5,
    "accel_noise_density_m_s2_rthz": 9e-5,
}
"""Constructor keywords differing from nominal in every parameter but the seed."""


def test_nominal_vehicle_is_the_specified_one():
    """Read from the class, as a consumer reads it without building a vehicle."""
    assert ImuSpacecraft.mass_kg == 750.0
    assert ImuSpacecraft.inertia_B_kg_m2 == (
        (900.0, 0.0, 0.0),
        (0.0, 800.0, 0.0),
        (0.0, 0.0, 600.0),
    )
    assert ImuSpacecraft.gyro_bias_P_rad_s == (1e-4, -2e-4, 5e-5)
    assert ImuSpacecraft.accel_bias_P_m_s2 == (1e-4, -5e-5, 2e-4)
    assert ImuSpacecraft.gyro_noise_density_rad_s_rthz == 5e-5
    assert ImuSpacecraft.accel_noise_density_m_s2_rthz == 5e-5
    assert ImuSpacecraft.noise_seed == 0
    assert ImuSpacecraft.gyro_bias_std_P_rad_s == 1e-4
    assert ImuSpacecraft.accel_bias_std_P_m_s2 == 1e-4


def test_each_keyword_defaults_to_its_class_attribute():
    """So a vehicle built with no keywords is the vehicle a consumer reads from the class."""
    parameters = inspect.signature(ImuSpacecraft).parameters
    assert len(parameters) == 7
    vehicle = ImuSpacecraft()
    for name, parameter in parameters.items():
        assert parameter.default is None, name
        assert getattr(vehicle, name) == getattr(ImuSpacecraft, name), name


def test_a_subclass_attribute_is_the_value_its_vehicles_are_built_with():
    """A subclass's class attributes build its hub and IMU; an explicit keyword still wins."""

    class Heavy(ImuSpacecraft):
        """An `ImuSpacecraft` whose class attributes differ in mass and gyro bias.

        Attributes:
            mass_kg: Hub mass, in kilograms.
            gyro_bias_P_rad_s: Constant gyro bias in the IMU platform frame, in rad/s.
        """

        mass_kg = 1000.0
        gyro_bias_P_rad_s = (3e-4, -1e-4, 2e-4)

    vehicle = Heavy()
    assert vehicle.spacecraft.hub.mHub == 1000.0
    assert np.array_equal(np.array(vehicle.imu.senRotBias).flatten(), Heavy.gyro_bias_P_rad_s)
    assert Heavy(mass_kg=500.0).spacecraft.hub.mHub == 500.0


def test_keywords_reach_the_basilisk_objects():
    """Every constructor keyword reaches the hub and the IMU it configures."""
    vehicle = ImuSpacecraft(**DISPERSED, noise_seed=11)
    hub = vehicle.spacecraft.hub
    assert hub.mHub == 500.0
    assert np.array_equal(np.array(hub.IHubPntBc_B), np.array(DISPERSED["inertia_B_kg_m2"]))
    assert np.array_equal(np.array(hub.r_BcB_B).flatten(), np.zeros(3))

    imu = vehicle.imu
    assert np.array_equal(np.array(imu.senRotBias).flatten(), DISPERSED["gyro_bias_P_rad_s"])
    assert np.array_equal(np.array(imu.senTransBias).flatten(), DISPERSED["accel_bias_P_m_s2"])
    assert np.array_equal(np.array(imu.dcm_PB), np.eye(3))
    assert np.array_equal(np.array(imu.sensorPos_B).flatten(), np.zeros(3))
    assert imu.RNGSeed == 11
    assert vehicle.models() == [vehicle.spacecraft]
    assert vehicle.sensor_models(0.25) == [imu]
    assert np.allclose(np.array(imu.PMatrixGyro), 7e-5 / 0.5 * np.eye(3), rtol=1e-15)
    assert np.allclose(np.array(imu.PMatrixAccel), 9e-5 / 0.5 * np.eye(3), rtol=1e-15)


def test_dispersed_builds_the_vehicle_from_biases_drawn_in_order():
    """`dispersed` draws the gyro then the accelerometer bias and builds the IMU with them."""
    vehicle, draws = ImuSpacecraft.dispersed(np.random.default_rng(7), noise_seed=11)
    rng = np.random.default_rng(7)
    gyro = np.asarray(ImuSpacecraft.gyro_bias_P_rad_s) + rng.normal(0.0, 1e-4, size=3)
    accel = np.asarray(ImuSpacecraft.accel_bias_P_m_s2) + rng.normal(0.0, 1e-4, size=3)
    assert list(draws) == ["gyro_bias_P_rad_s", "accel_bias_P_m_s2"]
    assert np.array_equal(draws["gyro_bias_P_rad_s"], gyro)
    assert np.array_equal(draws["accel_bias_P_m_s2"], accel)
    assert np.array_equal(np.array(vehicle.imu.senRotBias).flatten(), gyro)
    assert np.array_equal(np.array(vehicle.imu.senTransBias).flatten(), accel)
    assert vehicle.imu.RNGSeed == 11


def test_imu_reader_is_subscribed_to_the_imu_output():
    """The `imu` reader reads the IMU's own output message."""
    scenario = Leo500()
    reader = scenario.vehicle.readers()["imu"]
    assert reader.isLinked()
    datasets = scenario.run(_Guidance(), {}, duration_s=20.0)
    gyro, _ = datasets["imu/gyro_P"]
    assert np.array_equal(reader().AngVelPlatform, gyro[-1])


def test_recorded_fields_name_the_run_file_datasets_with_their_units():
    """`recorded_fields` names each run-file dataset with its units."""
    fields = ImuSpacecraft().recorded_fields()
    assert {name: units for name, (_, _, units) in fields.items()} == {
        "truth/sigma_BN": "MRP",
        "truth/omega_BN_B": "rad/s",
        "truth/r_N": "m",
        "truth/v_N": "m/s",
        "imu/gyro_P": "rad/s",
        "imu/accel_P": "m/s^2",
    }


class _Quiet(ImuSpacecraft):
    """An `ImuSpacecraft` without biases and with 1e-3 noise densities, so its output is its noise.

    Attributes:
        gyro_bias_P_rad_s: Constant gyro bias in the IMU platform frame, in rad/s.
        accel_bias_P_m_s2: Constant accelerometer bias in the IMU platform frame, in m/s^2.
        gyro_noise_density_rad_s_rthz: Gyro white-noise density, in rad/s/sqrt(Hz).
        accel_noise_density_m_s2_rthz: Accelerometer white-noise density, in m/s^2/sqrt(Hz).
    """

    gyro_bias_P_rad_s = (0.0, 0.0, 0.0)
    accel_bias_P_m_s2 = (0.0, 0.0, 0.0)
    gyro_noise_density_rad_s_rthz = 1e-3
    accel_noise_density_m_s2_rthz = 1e-3


@pytest.mark.parametrize("period_s", [1.0, 0.25])
def test_the_noise_per_sample_is_the_density_over_the_root_of_the_sample_period(period_s: float):
    """Each sample carries white noise of 1-sigma density / sqrt(period), whatever the period.

    The IMU sits at the centre of mass in free fall, so its accelerometer senses only noise, and
    its gyro senses the body rate plus noise.

    Args:
        period_s: The flight software's period, and so the IMU's sample period, in seconds.
    """
    scenario = Leo500(vehicle_type=_Quiet)
    scenario.record_s = period_s
    fsw = type("_Paced", (_Guidance,), {"period_s": period_s})()
    datasets = scenario.run(fsw, {}, duration_s=2000 * period_s)
    expected = 1e-3 / np.sqrt(period_s)
    gyro_noise = datasets["imu/gyro_P"][0] - datasets["truth/omega_BN_B"][0]
    for noise in (gyro_noise, datasets["imu/accel_P"][0]):
        assert np.std(noise[1:]) == pytest.approx(expected, rel=0.05)
        assert abs(np.corrcoef(noise[1:-1, 0], noise[2:, 0])[0, 1]) < 0.1
