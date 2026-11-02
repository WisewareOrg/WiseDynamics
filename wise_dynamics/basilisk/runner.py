"""Runs one case: a plant, one flight-software module, recorders, and the recorded datasets."""

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
from Basilisk.utilities import macros

from wise_dynamics.basilisk.plant import Message, Plant, RecordedField, SysModel
from wise_dynamics.run_file import Dataset

__all__ = ["Recording", "run_case"]

# Basilisk's SWIG-generated recorder classes are typed as `Any`.
type Recorder = Any

_NANOSECONDS_PER_SECOND = 1e9


class Recording(Mapping[str, Recorder]):
    """The recorders of one case, by message role, all sampled on one cadence from t = 0.

    Indexing by role gives the Basilisk recorder, whose payload fields are attributes and whose
    sample times in nanoseconds come from `times()`.
    """

    def __init__(self, recorders: Mapping[str, Recorder]) -> None:
        """Wrap recorders that have already run.

        Args:
            recorders: Message role to its recorder, at least one.
        """
        self._recorders = dict(recorders)

    def __getitem__(self, role: str) -> Recorder:
        return self._recorders[role]

    def __iter__(self) -> Iterator[str]:
        return iter(self._recorders)

    def __len__(self) -> int:
        return len(self._recorders)

    def datasets(self, fields: Mapping[str, RecordedField]) -> dict[str, Dataset]:
        """Recorded fields as unit-tagged datasets, ready for a run file.

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
        datasets = {"t_s": Dataset(np.asarray(first.times()) / _NANOSECONDS_PER_SECOND, "s")}
        for name, recorded in fields.items():
            recorder = self._recorders[recorded.role]
            try:
                data = np.array(getattr(recorder, recorded.field))
            except AttributeError:
                raise ValueError(
                    f"dataset {name!r}: the {recorded.role!r} recorder has no field "
                    f"{recorded.field!r}"
                ) from None
            datasets[name] = Dataset(data, recorded.units)
        return datasets


def run_case(
    plant: Plant[Any],
    fsw_module: SysModel,
    *,
    fsw_outputs: Mapping[str, Message],
    duration_s: float,
    record_s: float = 10.0,
) -> Recording:
    """Run a plant with one flight-software module and record it.

    Adds `fsw_module` to the plant's task after the plant's own models, records every message
    of `plant.truth_messages()` and of `fsw_outputs` every `record_s` from t = 0, and runs the
    plant for `duration_s`. The arguments are checked before the plant is changed. The plant
    runs once, so a second case needs a new plant.

    Args:
        plant: The plant to run.
        fsw_module: Any Basilisk module, for example one reading `plant.readers()`.
        fsw_outputs: Role to output message of `fsw_module` to record. Roles are distinct from
            the plant's truth roles.
        duration_s: Simulated time to run, in seconds.
        record_s: Recording period, in seconds.

    Returns:
        The `Recording`, holding the truth roles and the `fsw_outputs` roles.

    Raises:
        ValueError: An `fsw_outputs` role equals a truth role, or `record_s` or `duration_s` is
            not positive.
        RuntimeError: The plant has already run.
    """
    truth_messages = plant.truth_messages()
    collisions = sorted(set(truth_messages) & set(fsw_outputs))
    if collisions:
        raise ValueError(f"fsw_outputs reuses the plant's truth roles {collisions}")
    if not record_s > 0:
        raise ValueError(f"recording period must be positive, got {record_s} s")
    if not duration_s > 0:
        raise ValueError(f"duration must be positive, got {duration_s} s")

    plant.add_model(fsw_module)
    period_ns = macros.sec2nano(record_s)
    recorders = {
        role: message.recorder(period_ns)
        for role, message in {**truth_messages, **fsw_outputs}.items()
    }
    for recorder in recorders.values():
        plant.add_model(recorder)
    plant.execute(duration_s)
    return Recording(recorders)
