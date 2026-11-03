"""`Vehicle.write_run_file`: one HDF5 file, read back with h5py directly."""

import h5py
import numpy as np
import pytest

from wise_dynamics.basilisk import ImuSpacecraft


def test_datasets_and_attrs_round_trip_as_float64_with_units(tmp_path):
    path = tmp_path / "session" / "nested" / "run_0000.h5"
    ImuSpacecraft().write_run_file(
        path,
        {
            "t_s": (np.arange(3), "s"),
            "truth/r_N": (np.ones((3, 3)) * 7e6, "m"),
        },
        {"case": 0, "draw_gyro_bias_P_rad_s": np.array([0.1, 0.2, -0.3])},
    )

    with h5py.File(path, "r") as written:
        assert written["t_s"].dtype == np.float64
        assert np.array_equal(written["t_s"][:], [0.0, 1.0, 2.0])
        assert written["t_s"].attrs["units"] == "s"
        assert written["truth/r_N"].shape == (3, 3)
        assert written["truth/r_N"].attrs["units"] == "m"
        assert written.attrs["case"] == 0
        assert np.array_equal(written.attrs["draw_gyro_bias_P_rad_s"], [0.1, 0.2, -0.3])


def test_a_run_file_is_written_once(tmp_path):
    vehicle = ImuSpacecraft()
    vehicle.write_run_file(tmp_path / "run_0000.h5", {"t_s": (np.zeros(1), "s")})
    with pytest.raises(FileExistsError):
        vehicle.write_run_file(tmp_path / "run_0000.h5", {"t_s": (np.zeros(1), "s")})


def test_a_dataset_without_units_is_refused(tmp_path):
    path = tmp_path / "run_0000.h5"
    with pytest.raises(ValueError, match="dataset 't_s' has an empty units string"):
        ImuSpacecraft().write_run_file(path, {"t_s": (np.zeros(1), "")})
    assert not path.exists()


def test_a_failed_write_leaves_no_file(tmp_path):
    vehicle = ImuSpacecraft()
    path = tmp_path / "run_0000.h5"
    with pytest.raises(TypeError):
        vehicle.write_run_file(path, {"t_s": (np.zeros(1), "s")}, {"bad": None})
    assert not path.exists()
    vehicle.write_run_file(path, {"t_s": (np.zeros(1), "s")})
