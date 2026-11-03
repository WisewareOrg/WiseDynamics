"""`RunFile`: one HDF5 file, read back with h5py directly."""

import h5py
import numpy as np
import pytest

from wise_dynamics import RunFile


def test_datasets_and_attrs_round_trip_as_float64_with_units(tmp_path):
    path = tmp_path / "session" / "nested" / "run_0000.h5"
    run_file = RunFile(path, {"case": 0})
    run_file.write(
        {
            "t_s": (np.arange(3), "s"),
            "truth/r_N": (np.ones((3, 3)) * 7e6, "m"),
        },
        {"draw_sigma0_BN": np.array([0.1, 0.2, -0.3])},
    )

    assert run_file.path == path
    with h5py.File(path, "r") as written:
        assert written["t_s"].dtype == np.float64
        assert np.array_equal(written["t_s"][:], [0.0, 1.0, 2.0])
        assert written["t_s"].attrs["units"] == "s"
        assert written["truth/r_N"].shape == (3, 3)
        assert written["truth/r_N"].attrs["units"] == "m"
        assert written.attrs["case"] == 0
        assert np.array_equal(written.attrs["draw_sigma0_BN"], [0.1, 0.2, -0.3])


def test_a_run_file_is_written_once(tmp_path):
    run_file = RunFile(tmp_path / "run_0000.h5")
    run_file.write({"t_s": (np.zeros(1), "s")})
    with pytest.raises(FileExistsError):
        run_file.write({"t_s": (np.zeros(1), "s")})


def test_attrs_cannot_replace_the_files_own(tmp_path):
    run_file = RunFile(tmp_path / "run_0000.h5", {"seed": 0, "case": 0})
    with pytest.raises(ValueError, match=r"own attrs \['seed'\]"):
        run_file.write({"t_s": (np.zeros(1), "s")}, {"seed": 1})
    assert not run_file.path.exists()


def test_a_dataset_without_units_is_refused(tmp_path):
    with pytest.raises(ValueError, match="t_s"):
        RunFile(tmp_path / "run_0000.h5").write({"t_s": (np.zeros(1), "")})


def test_a_failed_write_leaves_no_file(tmp_path):
    run_file = RunFile(tmp_path / "run_0000.h5")
    with pytest.raises(TypeError):
        run_file.write({"t_s": (np.zeros(1), "s")}, {"bad": None})
    assert not run_file.path.exists()
    run_file.write({"t_s": (np.zeros(1), "s")})
