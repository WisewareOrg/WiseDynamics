"""The plumbing every Basilisk vehicle shares: the run file it writes."""

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = ["Vehicle"]


class Vehicle:
    """The plumbing every vehicle shares: writing its run file.

    A subclass builds its Basilisk objects and provides `models()`, the propagated modules a
    scenario steps with the dynamics, `sensor_models(sample_period_s)`, the sensors it samples at
    the flight software's period, `readers()`, its sensor outputs by role, and
    `recorded_fields()`, run-file dataset path to the `(message, field, units)` it records; a
    scenario places its `spacecraft`.
    """

    def write_run_file(
        self,
        path: str | PathLike[str],
        datasets: Mapping[str, tuple[Any, str]],
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """Write one HDF5 run file.

        A write that fails part-way removes the file it created and re-raises.

        Args:
            path: The file to create. Its parent directories are created.
            datasets: Dataset name to `(array, units)`, as `Scenario.run` returns them. A name may
                contain `/` to place the dataset in a group, for example `"truth/r_N"`. Each array
                is written as float64 with a string attribute `units`.
            attrs: Root attributes, name to value.

        Raises:
            BaseException: Whatever the write raised, re-raised once the partial file is removed:
                `FileExistsError` when `path` already exists, and `TypeError` when an attribute or
                dataset value is of a type HDF5 cannot store.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        run_file = h5py.File(path, "w-")
        try:
            with run_file:
                for name, value in (attrs or {}).items():
                    run_file.attrs[name] = value
                for name, (data, units) in datasets.items():
                    dataset = run_file.create_dataset(name, data=np.asarray(data, dtype=np.float64))
                    dataset.attrs["units"] = units
        except BaseException:
            path.unlink(missing_ok=True)
            raise
