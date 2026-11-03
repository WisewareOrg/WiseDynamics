"""The stable interface a consumer's host adapter and run-file reader code against."""

import inspect

import pytest

import wise_dynamics
import wise_dynamics.basilisk
from tests.support import EARTH_MU_M3_S2
from wise_dynamics.basilisk import PLANTS, ImuSpacecraft, Plant


def test_imu_spacecraft_is_the_registered_plant_and_implements_the_interface():
    assert PLANTS["imu_spacecraft"] is ImuSpacecraft
    assert issubclass(ImuSpacecraft, Plant)
    assert not inspect.isabstract(ImuSpacecraft)


def test_readers_has_role_imu_and_truth_messages_has_roles_truth_and_imu():
    plant = ImuSpacecraft()
    assert set(plant.readers()) == {"imu"}
    assert set(plant.truth_messages()) == {"truth", "imu"}


def test_initial_state_returns_exactly_its_documented_keys():
    assert set(ImuSpacecraft().initial_state()) == {"sigma0_BN", "r0_N", "v0_N", "omega0_BN_B"}


def test_mu_is_earth_point_mass():
    assert ImuSpacecraft.mu == EARTH_MU_M3_S2


def test_recorded_fields_name_the_run_file_datasets_with_their_units():
    assert {name: field.units for name, field in ImuSpacecraft.recorded_fields.items()} == {
        "truth/sigma_BN": "MRP",
        "truth/omega_BN_B": "rad/s",
        "truth/r_N": "m",
        "truth/v_N": "m/s",
        "imu/gyro_P": "rad/s",
        "imu/accel_P": "m/s^2",
    }


@pytest.mark.parametrize(
    ("package", "names"),
    [
        (
            wise_dynamics,
            {"Dispersion", "write_run"},
        ),
        (
            wise_dynamics.basilisk,
            {
                "PLANTS",
                "ImuSpacecraft",
                "ImuSpacecraftConfig",
                "OrbitElements",
                "Plant",
                "RecordedField",
                "Recording",
                "RigidHub",
                "run_case",
            },
        ),
    ],
    ids=["wise_dynamics", "wise_dynamics.basilisk"],
)
def test_each_package_exports_exactly_its_public_names(package, names):
    assert set(package.__all__) == names
    for name in names:
        assert getattr(package, name) is not None, name
