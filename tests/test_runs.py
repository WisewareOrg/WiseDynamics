"""Checks `write_run` writes the §2 HDF5 layout: root attrs, and datasets each carrying their
own `units` attribute, read back with h5py directly rather than trusting the writer's own
claims."""

import h5py
import numpy as np

from sim.basilisk.runs import write_run


def test_write_run_round_trips_through_h5py(tmp_path):
    arrays = {
        "t_s": (np.array([0.0, 10.0, 20.0]), "s"),
        "truth/sigma_BN": (np.zeros((3, 3)), "MRP"),
        "imu/gyro_P": (np.ones((3, 3)) * 1e-4, "rad/s"),
    }
    attrs = {"case": 0, "seed": 0, "producer": "test", "draw_sigma0_BN": [0.1, 0.2, -0.3]}

    path = tmp_path / "run_0000.h5"
    write_run(path, arrays, attrs)

    with h5py.File(path, "r") as f:
        assert f.attrs["case"] == 0
        assert f.attrs["seed"] == 0
        assert f.attrs["producer"] == "test"
        assert np.allclose(f.attrs["draw_sigma0_BN"], [0.1, 0.2, -0.3])

        assert f["t_s"].shape == (3,)
        assert f["t_s"].dtype == np.float64
        assert f["t_s"].attrs["units"] == "s"

        assert f["truth/sigma_BN"].shape == (3, 3)
        assert f["truth/sigma_BN"].attrs["units"] == "MRP"

        assert f["imu/gyro_P"].attrs["units"] == "rad/s"
        assert np.allclose(f["imu/gyro_P"][:], 1e-4)
