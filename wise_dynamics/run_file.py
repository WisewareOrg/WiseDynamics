"""Writes one HDF5 file per simulated case: unit-tagged float64 datasets plus root attributes."""

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from wise_dynamics.dataset import Dataset

__all__ = ["write_run"]


def write_run(
    path: str | PathLike[str],
    datasets: Mapping[str, Dataset],
    attrs: Mapping[str, Any],
) -> Path:
    """Write one run file, creating its parent directories as needed.

    A write that fails part-way removes the file it created and re-raises.

    Args:
        path: The file to create. Its parent directories are created when absent.
        datasets: Dataset name to `Dataset`. A name may contain `/` to place the dataset in a
            group, for example `"truth/r_N"`. Each is written as float64 with a string attribute
            `units`. A plain `(array, units)` tuple is accepted in place of a `Dataset`.
        attrs: Root attribute name to value, written as given.

    Returns:
        The path written, as a `Path`.

    Raises:
        FileExistsError: `path` already exists; a run file is written exactly once.
        ValueError: A dataset has an empty units string.
        TypeError: An attribute or dataset value is of a type HDF5 cannot store.
    """
    path = Path(path)
    for name, (_, units) in datasets.items():
        if not units:
            raise ValueError(f"dataset {name!r} has an empty units string")

    path.parent.mkdir(parents=True, exist_ok=True)
    run_file = h5py.File(path, "w-")
    try:
        with run_file:
            for name, value in attrs.items():
                run_file.attrs[name] = value
            for name, (data, units) in datasets.items():
                dataset = run_file.create_dataset(name, data=np.asarray(data, dtype=np.float64))
                dataset.attrs["units"] = units
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path
