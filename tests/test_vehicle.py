"""`Vehicle.write_run_file`: one HDF5 file, read back with h5py directly."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from wise_dynamics.basilisk import ImuSpacecraft


def test_datasets_and_attrs_round_trip_as_float64_with_units(tmp_path: Path):
    """Datasets read back as float64 with their units, and root attributes as written.

    Args:
        tmp_path: The test's own temporary directory.
    """
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


def test_a_run_file_is_written_once(tmp_path: Path):
    """A second write to the same path is refused.

    Args:
        tmp_path: The test's own temporary directory.
    """
    vehicle = ImuSpacecraft()
    vehicle.write_run_file(tmp_path / "run_0000.h5", {"t_s": (np.zeros(1), "s")})
    with pytest.raises(FileExistsError):
        vehicle.write_run_file(tmp_path / "run_0000.h5", {"t_s": (np.zeros(1), "s")})


def test_a_failed_write_leaves_no_file(tmp_path: Path):
    """A write that fails part-way removes its file, and the path stays writable.

    Args:
        tmp_path: The test's own temporary directory.
    """
    vehicle = ImuSpacecraft()
    path = tmp_path / "run_0000.h5"
    with pytest.raises(TypeError):
        vehicle.write_run_file(path, {"t_s": (np.zeros(1), "s")}, {"bad": None})
    assert not path.exists()
    vehicle.write_run_file(path, {"t_s": (np.zeros(1), "s")})


def test_string_datasets_round_trip_as_text_with_their_units(tmp_path: Path):
    """A string array is written as UTF-8 text and read back as the same strings, units kept.

    Args:
        tmp_path: The test's own temporary directory.
    """
    path = tmp_path / "run_0000.h5"
    ImuSpacecraft().write_run_file(
        path,
        {"tlm/state": (np.array(["idle", "propagated", ""]), "enum"), "t_s": (np.arange(3), "s")},
    )

    with h5py.File(path, "r") as written:
        assert h5py.check_string_dtype(written["tlm/state"].dtype) is not None
        assert written["tlm/state"].asstr()[:].tolist() == ["idle", "propagated", ""]
        assert written["tlm/state"].attrs["units"] == "enum"
        assert written["t_s"].dtype == np.float64
