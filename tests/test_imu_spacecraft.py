"""`ImuSpacecraft`: its nominal vehicle, how its keywords reach the Basilisk objects it builds, its
draws, its reader, and what it records."""

import inspect

import numpy as np
from Basilisk.fswAlgorithms import inertial3D

from wise_dynamics.basilisk import ImuSpacecraft, Leo500

DISPERSED = {
    "mass_kg": 500.0,
    "inertia_B_kg_m2": ((150.0, -1.0, 0.0), (-1.0, 200.0, 0.0), (0.0, 0.0, 250.0)),
    "gyro_bias_P_rad_s": (3e-4, -1e-4, 2e-4),
    "accel_bias_P_m_s2": (-2e-4, 4e-4, 1e-4),
    "gyro_noise_std_rad_s": 7e-5,
    "accel_noise_std_m_s2": 9e-5,
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
    assert ImuSpacecraft.gyro_noise_std_rad_s == 5e-5
    assert ImuSpacecraft.accel_noise_std_m_s2 == 5e-5
    assert ImuSpacecraft.noise_seed == 0
    assert ImuSpacecraft.standard_dispersion == {
        "gyro_bias_P_rad_s": 1e-4,
        "accel_bias_P_m_s2": 1e-4,
    }


def test_each_keyword_defaults_to_its_class_attribute():
    """So a vehicle built with no keywords is the vehicle a consumer reads from the class."""
    parameters = inspect.signature(ImuSpacecraft).parameters
    assert len(parameters) == 7
    vehicle = ImuSpacecraft()
    for name, parameter in parameters.items():
        assert parameter.default is None, name
        assert getattr(vehicle, name) == getattr(ImuSpacecraft, name), name


def test_a_subclass_attribute_is_the_value_its_vehicles_are_built_with():
    class Heavy(ImuSpacecraft):
        mass_kg = 1000.0
        gyro_bias_P_rad_s = (3e-4, -1e-4, 2e-4)

    vehicle = Heavy()
    assert vehicle.spacecraft.hub.mHub == 1000.0
    assert np.array_equal(np.array(vehicle.imu.senRotBias).flatten(), Heavy.gyro_bias_P_rad_s)
    assert Heavy(mass_kg=500.0).spacecraft.hub.mHub == 500.0


def test_keywords_reach_the_basilisk_objects():
    vehicle = ImuSpacecraft(**DISPERSED, noise_seed=11)
    hub = vehicle.spacecraft.hub
    assert hub.mHub == 500.0
    assert np.array_equal(np.array(hub.IHubPntBc_B), np.array(DISPERSED["inertia_B_kg_m2"]))
    assert np.array_equal(np.array(hub.r_BcB_B).flatten(), np.zeros(3))

    imu = vehicle.imu
    assert np.array_equal(np.array(imu.senRotBias).flatten(), DISPERSED["gyro_bias_P_rad_s"])
    assert np.array_equal(np.array(imu.senTransBias).flatten(), DISPERSED["accel_bias_P_m_s2"])
    assert np.array_equal(np.array(imu.PMatrixGyro), 7e-5 * np.eye(3))
    assert np.array_equal(np.array(imu.PMatrixAccel), 9e-5 * np.eye(3))
    assert np.array_equal(np.array(imu.dcm_PB), np.eye(3))
    assert np.array_equal(np.array(imu.sensorPos_B).flatten(), np.zeros(3))
    assert imu.RNGSeed == 11
    assert vehicle.models() == [vehicle.spacecraft, imu]


def test_draw_adds_normal_samples_to_the_defaults_in_dispersion_order():
    draws = ImuSpacecraft.draw(np.random.default_rng(7))
    rng = np.random.default_rng(7)
    assert list(draws) == list(ImuSpacecraft.standard_dispersion)
    for name, std in ImuSpacecraft.standard_dispersion.items():
        expected = np.asarray(getattr(ImuSpacecraft, name)) + rng.normal(0.0, std, size=3)
        assert np.array_equal(draws[name], expected), name


def test_imu_reader_is_subscribed_to_the_imu_output():
    vehicle = ImuSpacecraft()
    reader = vehicle.readers()["imu"]
    assert reader.isLinked()
    scenario = Leo500()
    scenario.add(vehicle)
    datasets = scenario.run(inertial3D.inertial3D(), fsw_fields={}, duration_s=20.0)
    gyro, _ = datasets["imu/gyro_P"]
    assert np.array_equal(reader().AngVelPlatform, gyro[-1])


def test_recorded_fields_name_the_run_file_datasets_with_their_units():
    fields = ImuSpacecraft().recorded_fields()
    assert {name: units for name, (_, _, units) in fields.items()} == {
        "truth/sigma_BN": "MRP",
        "truth/omega_BN_B": "rad/s",
        "truth/r_N": "m",
        "truth/v_N": "m/s",
        "imu/gyro_P": "rad/s",
        "imu/accel_P": "m/s^2",
    }
