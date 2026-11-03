"""One simulated case's HDF5 run file: unit-tagged float64 datasets plus root attributes."""

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = ["RunFile"]


class RunFile:
    """A run file at `path`, written once by `write`.

    Args:
        path: The file to create. Its parent directories are created when it is written.
        attrs: Root attributes the file carries, name to value, which `write` may not replace.

    Attributes:
        path: The file, as a `Path`.
    """

    def __init__(self, path: str | PathLike[str], attrs: Mapping[str, Any] | None = None) -> None:
        self.path = Path(path)
        self._attrs = dict(attrs or {})

    def write(
        self, datasets: Mapping[str, tuple[Any, str]], attrs: Mapping[str, Any] | None = None
    ) -> None:
        """Write the file. A write that fails part-way removes the file it created and re-raises.

        Args:
            datasets: Dataset name to `(array, units)`, as `run_case` returns them. A name may
                contain `/` to place the dataset in a group, for example `"truth/r_N"`. Each array
                is written as float64 with a string attribute `units`.
            attrs: Further root attributes, name to value, written beside the file's own.

        Raises:
            FileExistsError: `path` already exists.
            ValueError: A dataset has an empty units string, or `attrs` names one of the file's
                own attributes.
            TypeError: An attribute or dataset value is of a type HDF5 cannot store.
        """
        attrs = attrs or {}
        reserved = sorted(set(self._attrs) & set(attrs))
        if reserved:
            raise ValueError(f"attrs may not replace the run file's own attrs {reserved}")
        for name, (_, units) in datasets.items():
            if not units:
                raise ValueError(f"dataset {name!r} has an empty units string")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        run_file = h5py.File(self.path, "w-")
        try:
            with run_file:
                for name, value in {**self._attrs, **attrs}.items():
                    run_file.attrs[name] = value
                for name, (data, units) in datasets.items():
                    dataset = run_file.create_dataset(name, data=np.asarray(data, dtype=np.float64))
                    dataset.attrs["units"] = units
        except BaseException:
            self.path.unlink(missing_ok=True)
            raise
