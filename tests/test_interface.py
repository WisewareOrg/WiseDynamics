"""The stable interface a consumer's host adapter and run-file reader code against."""

import inspect

from wise_dynamics.basilisk import PLANTS, ImuSpacecraft, Plant

# Earth's mu in Basilisk 2.12.1's gravBodyFactory().createEarth(), m^3/s^2.
EARTH_MU_M3_S2 = 3.986004360e14


def test_imu_spacecraft_is_the_registered_plant_and_implements_the_contract():
    assert PLANTS["imu_spacecraft"] is ImuSpacecraft
    assert issubclass(ImuSpacecraft, Plant)
    assert not inspect.isabstract(ImuSpacecraft)


def test_readers_has_role_imu_and_truth_messages_has_roles_truth_and_imu():
    plant = ImuSpacecraft()
    assert set(plant.readers()) == {"imu"}
    assert set(plant.truth_messages()) == {"truth", "imu"}


def test_initial_state_returns_exactly_the_settled_keys():
    assert set(ImuSpacecraft().initial_state()) == {"sigma0_BN", "r0_N", "v0_N", "omega0_BN_B"}


def test_mu_is_earth_point_mass_on_the_class_and_the_plant():
    assert ImuSpacecraft.mu == EARTH_MU_M3_S2
    assert ImuSpacecraft().mu == EARTH_MU_M3_S2


def test_recorded_fields_name_the_run_file_datasets_with_their_units():
    assert {name: field.units for name, field in ImuSpacecraft.recorded_fields.items()} == {
        "truth/sigma_BN": "MRP",
        "truth/omega_BN_B": "rad/s",
        "truth/r_N": "m",
        "truth/v_N": "m/s",
        "imu/gyro_P": "rad/s",
        "imu/accel_P": "m/s^2",
    }
