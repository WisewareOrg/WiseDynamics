"""The type of the `sim_run_writer` fixture's value."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from wise_dynamics.dataset import Dataset

__all__ = ["RunWriter"]


class RunWriter(Protocol):
    """The `sim_run_writer` fixture's value: writes the case's run file."""

    def __call__(
        self, datasets: Mapping[str, Dataset], extra_attrs: Mapping[str, Any] | None = None
    ) -> Path:
        """Write `<out_dir>/run_<NNNN>.h5` and return its path.

        Args:
            datasets: Dataset name to `Dataset`, as `wise_dynamics.write_run` takes them.
            extra_attrs: Further root attributes, for example the flight software's version as
                `versions_<distribution>`.

        Raises:
            ValueError: An entry of `extra_attrs` names one of the case's own root attributes.
            FileExistsError: The run file already exists.
        """
        ...
