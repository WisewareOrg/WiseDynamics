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
    scenario steps with the dynamics; `sensor_models(sample_period_s)`, which sets each sensor's
    per-sample noise for that period and lists the sensors a scenario samples at it; `readers()`,
    its sensor outputs by role; `recorded_fields()`, run-file dataset path to the
    `(message, field, units)` it records; and `disperse(rng, *, noise_seed)`, which draws and
    applies its own dispersion in place. It may override `writers()`, its actuator inputs by role,
    and `finish(datasets)` to change what the recorders captured before the run returns it. A
    scenario places its `spacecraft`.

    Attributes:
        spacecraft: The Basilisk `Spacecraft` hub, which a subclass builds.
    """

    spacecraft: Any

    def disperse(self, rng: np.random.Generator, *, noise_seed: int) -> dict[str, np.ndarray]:
        """Draw this vehicle's dispersion from `rng`, apply it to this vehicle and seed its sensor noise.

        Args:
            rng: The generator the dispersion is drawn from.
            noise_seed: The seed of this vehicle's sensor noise.

        Returns:
            What was drawn, name to value.

        Raises:
            NotImplementedError: Always: a subclass provides it.
        """
        raise NotImplementedError

    def models(self) -> list[Any]:
        """List the vehicle's propagated Basilisk modules in task order.

        Returns:
            The modules a scenario steps with the dynamics.

        Raises:
            NotImplementedError: Always: a subclass provides it.
        """
        raise NotImplementedError

    def sensor_models(self, sample_period_s: float) -> list[Any]:
        """Set each sensor's per-sample noise for its sample period, and list the sensors.

        Args:
            sample_period_s: The period the sensors sample at, in seconds.

        Returns:
            The sensors a scenario samples at that period.

        Raises:
            NotImplementedError: Always: a subclass provides it.
        """
        raise NotImplementedError

    def readers(self) -> dict[str, Any]:
        """Build the vehicle's sensor readers.

        Returns:
            Role to the Basilisk message reader subscribed to that sensor's output.

        Raises:
            NotImplementedError: Always: a subclass provides it.
        """
        raise NotImplementedError

    def recorded_fields(self) -> dict[str, tuple[Any, str, str]]:
        """Declare what the vehicle records.

        Returns:
            Run-file dataset path to the `(message, field, units)` it records.

        Raises:
            NotImplementedError: Always: a subclass provides it.
        """
        raise NotImplementedError

    def writers(self) -> dict[str, Any]:
        """The vehicle's actuator inputs by role: by default it has none.

        Returns:
            Role to the stand-alone Basilisk message its device subscribes to, the same object at
            every call; `{}` here.
        """
        return {}

    def finish(self, datasets: Mapping[str, tuple[Any, str]]) -> dict[str, tuple[Any, str]]:
        """Turn what the recorders captured into what the run file holds.

        Args:
            datasets: Dataset name to `(array, units)`, as the recorders captured them.

        Returns:
            The datasets the run returns; by default the same ones.
        """
        return dict(datasets)

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
                is written as float64, or as variable-length UTF-8 strings when it holds
                strings, with a string attribute `units`.
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
                    array = np.asarray(data)
                    if array.dtype.kind == "U":
                        dataset = run_file.create_dataset(name, data=array.astype(object), dtype=h5py.string_dtype())
                    else:
                        dataset = run_file.create_dataset(name, data=array.astype(np.float64))
                    dataset.attrs["units"] = units
        except BaseException:
            path.unlink(missing_ok=True)
            raise
