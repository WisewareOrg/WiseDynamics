"""The interface every Basilisk vehicle meets, and the run file each vehicle writes."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np

__all__ = ["Vehicle"]


class Vehicle(ABC):
    """A spacecraft's mass properties, sensors and logging, independent of where it flies.

    A subclass takes its parameters as keywords that default to `None`, meaning the class
    attribute of the same name read when the vehicle is built. It builds its Basilisk objects and
    lists them in `models()`; a `Scenario` adds them to its task and places `spacecraft`. Every
    vehicle takes `noise_seed: int`, the seed of its sensor noise, which the pytest plugin sets for
    each case.

    Attributes:
        standard_dispersion: Keyword to its per-axis 1-sigma, drawn in this order by `draw`.
        spacecraft: The Basilisk `Spacecraft` hub a scenario places and attaches gravity to.
    """

    standard_dispersion: ClassVar[Mapping[str, float]]
    spacecraft: Any

    @classmethod
    def draw(cls, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Overrides drawn from `rng`: each `standard_dispersion` entry, in order, as the class
        attribute of that name plus independent normal samples of that 1-sigma."""
        draws = {}
        for name, std in cls.standard_dispersion.items():
            nominal = np.asarray(getattr(cls, name), dtype=float)
            draws[name] = nominal + rng.normal(0.0, std, size=nominal.shape)
        return draws

    @abstractmethod
    def models(self) -> list[Any]:
        """The vehicle's Basilisk modules, in the order a scenario adds them to its task."""

    @abstractmethod
    def readers(self) -> dict[str, Any]:
        """The vehicle's sensor outputs by role, each a Basilisk message reader subscribed to its
        sensor, for a flight-software module to read, built afresh on each call."""

    @abstractmethod
    def recorded_fields(self) -> dict[str, tuple[Any, str, str]]:
        """Run-file dataset path to the `(message, field, units)` it records from this vehicle."""

    def write_run_file(
        self,
        path: str | PathLike[str],
        datasets: Mapping[str, tuple[Any, str]],
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        """Write one HDF5 run file. A write that fails part-way removes the file it created and
        re-raises.

        Args:
            path: The file to create. Its parent directories are created.
            datasets: Dataset name to `(array, units)`, as `Scenario.run` returns them. A name may
                contain `/` to place the dataset in a group, for example `"truth/r_N"`. Each array
                is written as float64 with a string attribute `units`.
            attrs: Root attributes, name to value.

        Raises:
            FileExistsError: `path` already exists.
            ValueError: A dataset has an empty units string.
            TypeError: An attribute or dataset value is of a type HDF5 cannot store.
        """
        for name, (_, units) in datasets.items():
            if not units:
                raise ValueError(f"dataset {name!r} has an empty units string")

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
