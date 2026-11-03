"""The recorders of one case, and the run-file datasets they hold."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from wise_dynamics.basilisk.recorded_field import RecordedField

__all__ = ["Recording"]

_NANOSECONDS_PER_SECOND = 1e9


class Recording:
    """The Basilisk recorders of one case, by message role, all sampled on one cadence from
    t = 0."""

    def __init__(self, recorders: Mapping[str, Any]) -> None:
        """Wrap recorders that have already run.

        Args:
            recorders: Message role to its recorder, at least one.

        Raises:
            ValueError: `recorders` is empty.
        """
        if not recorders:
            raise ValueError("a Recording needs at least one recorder")
        self._recorders = dict(recorders)

    def datasets(self, fields: Mapping[str, RecordedField]) -> dict[str, tuple[np.ndarray, str]]:
        """Recorded fields as `(array, units)` datasets, ready for a run file.

        Args:
            fields: Dataset name to the recorded field it holds, for example a plant's
                `recorded_fields` merged with the flight software's own.

        Returns:
            `t_s`, the sample times in seconds as shape `(N,)`, followed by one dataset per
            entry of `fields`, each of `N` samples.

        Raises:
            KeyError: A field's role is absent from this recording.
            ValueError: A field is absent from its recorder.
        """
        first = next(iter(self._recorders.values()))
        datasets = {"t_s": (np.asarray(first.times()) / _NANOSECONDS_PER_SECOND, "s")}
        for name, recorded in fields.items():
            recorder = self._recorders[recorded.role]
            try:
                data = np.array(getattr(recorder, recorded.field))
            except AttributeError:
                raise ValueError(
                    f"dataset {name!r}: the {recorded.role!r} recorder has no field "
                    f"{recorded.field!r}"
                ) from None
            datasets[name] = (data, recorded.units)
        return datasets
