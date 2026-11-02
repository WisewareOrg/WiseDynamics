"""`write_run`: one HDF5 file, read back with h5py directly."""

import h5py
import numpy as np
import pytest

from wise_dynamics import Dataset, write_run


def test_datasets_and_attrs_round_trip_as_float64_with_units(tmp_path):
    path = tmp_path / "campaign" / "nested" / "run_0000.h5"
    written = write_run(
        path,
        {
            "t_s": Dataset(np.arange(3), "s"),
            "truth/r_N": (np.ones((3, 3)) * 7e6, "m"),
        },
        {"case": 0, "draw_sigma0_BN": np.array([0.1, 0.2, -0.3])},
    )

    assert written == path
    with h5py.File(path, "r") as run_file:
        assert run_file["t_s"].dtype == np.float64
        assert np.array_equal(run_file["t_s"][:], [0.0, 1.0, 2.0])
        assert run_file["t_s"].attrs["units"] == "s"
        assert run_file["truth/r_N"].shape == (3, 3)
        assert run_file["truth/r_N"].attrs["units"] == "m"
        assert run_file.attrs["case"] == 0
        assert np.array_equal(run_file.attrs["draw_sigma0_BN"], [0.1, 0.2, -0.3])


def test_a_run_file_is_written_once(tmp_path):
    path = tmp_path / "run_0000.h5"
    write_run(path, {"t_s": Dataset(np.zeros(1), "s")}, {})
    with pytest.raises(FileExistsError):
        write_run(path, {"t_s": Dataset(np.zeros(1), "s")}, {})


def test_a_dataset_without_units_is_refused(tmp_path):
    with pytest.raises(ValueError, match="t_s"):
        write_run(tmp_path / "run_0000.h5", {"t_s": Dataset(np.zeros(1), "")}, {})


def test_a_failed_write_leaves_no_file(tmp_path):
    path = tmp_path / "run_0000.h5"
    with pytest.raises(TypeError):
        write_run(path, {"t_s": Dataset(np.zeros(1), "s")}, {"bad": None})
    assert not path.exists()
    write_run(path, {"t_s": Dataset(np.zeros(1), "s")}, {})
