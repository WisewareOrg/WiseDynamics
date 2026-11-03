"""Runs one case: a plant, one flight-software module and a recorder per recorded field."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from Basilisk.utilities import macros

from wise_dynamics.basilisk.plant import Plant

__all__ = ["run_case"]

_NANOSECONDS_PER_SECOND = 1e9


def run_case(
    plant: Plant,
    fsw_module: Any,
    *,
    fields: Mapping[str, tuple[Any, str, str]],
    duration_s: float,
    record_s: float = 10.0,
) -> dict[str, tuple[np.ndarray, str]]:
    """Run a plant with one flight-software module and record the given fields.

    Adds `fsw_module` to the plant's task after the plant's own models, attaches one Basilisk
    recorder per entry of `fields` sampling every `record_s` from t = 0, and runs the plant for
    `duration_s`. The arguments are checked before the plant is changed. The plant runs once, so
    a second case needs a new plant.

    Args:
        plant: The plant to run.
        fsw_module: Any Basilisk module, for example one reading `plant.readers()`.
        fields: Run-file dataset path to the `(message, field, units)` it records, for example
            `plant.recorded_fields()` merged with the flight software's own outputs.
        duration_s: Simulated time to run, in seconds.
        record_s: Recording period, in seconds.

    Returns:
        Dataset path to `(array, units)`: `t_s`, the sample times in seconds as shape `(N,)`,
        then one entry per entry of `fields`, each of `N` samples.

    Raises:
        ValueError: `fields` is empty, a field is absent from its message, or `record_s` or
            `duration_s` is not positive.
        RuntimeError: The plant has already run.
    """
    if not fields:
        raise ValueError("run_case needs at least one field to record")
    if not record_s > 0:
        raise ValueError(f"recording period must be positive, got {record_s} s")
    if not duration_s > 0:
        raise ValueError(f"duration must be positive, got {duration_s} s")

    plant.add_model(fsw_module)
    period_ns = macros.sec2nano(record_s)
    recorders = {path: message.recorder(period_ns) for path, (message, _, _) in fields.items()}
    for recorder in recorders.values():
        plant.add_model(recorder)
    plant.execute(duration_s)

    first = next(iter(recorders.values()))
    datasets = {"t_s": (np.asarray(first.times()) / _NANOSECONDS_PER_SECOND, "s")}
    for path, (_, field, units) in fields.items():
        try:
            datasets[path] = (np.array(getattr(recorders[path], field)), units)
        except AttributeError:
            raise ValueError(f"dataset {path!r}: its message has no field {field!r}") from None
    return datasets
