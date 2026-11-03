"""Runs one case: a vehicle, one flight-software module and a recorder per recorded field."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from Basilisk.utilities import macros

from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["run_case"]

_NANOSECONDS_PER_SECOND = 1e9


def run_case(
    vehicle: Vehicle,
    fsw_module: Any,
    *,
    fields: Mapping[str, tuple[Any, str, str]],
    duration_s: float,
    record_s: float = 10.0,
) -> dict[str, tuple[np.ndarray, str]]:
    """Run a vehicle with one flight-software module and record the given fields.

    Adds `fsw_module` to the vehicle's task after the vehicle's own models, attaches one Basilisk
    recorder per entry of `fields` sampling every `record_s` from t = 0, and runs the vehicle for
    `duration_s`. The arguments are checked before the vehicle is changed.

    Args:
        vehicle: The vehicle to run.
        fsw_module: Any Basilisk module, for example one reading `vehicle.readers()`.
        fields: Run-file dataset path to the `(message, field, units)` it records, for example
            `vehicle.recorded_fields()` merged with the flight software's own outputs.
        duration_s: Simulated time to run, in seconds.
        record_s: Recording period, in seconds.

    Returns:
        Dataset path to `(array, units)`: `t_s`, the sample times in seconds as shape `(N,)`,
        then one entry per entry of `fields`, each of `N` samples.

    Raises:
        ValueError: `fields` is empty, a field is absent from its message, or `record_s` or
            `duration_s` is not positive.
        RuntimeError: The vehicle has already run.
    """
    if not fields:
        raise ValueError("run_case needs at least one field to record")
    if not record_s > 0:
        raise ValueError(f"recording period must be positive, got {record_s} s")
    if not duration_s > 0:
        raise ValueError(f"duration must be positive, got {duration_s} s")

    period_ns = macros.sec2nano(record_s)
    recorders = {path: message.recorder(period_ns) for path, (message, _, _) in fields.items()}
    for path, (_, field, _) in fields.items():
        if not isinstance(getattr(type(recorders[path]), field, None), property):
            raise ValueError(f"dataset {path!r}: its message has no field {field!r}")

    vehicle.add_model(fsw_module)
    for recorder in recorders.values():
        vehicle.add_model(recorder)
    vehicle.execute(duration_s)

    first = next(iter(recorders.values()))
    datasets = {"t_s": (np.asarray(first.times()) / _NANOSECONDS_PER_SECOND, "s")}
    for path, (_, field, units) in fields.items():
        datasets[path] = (np.array(getattr(recorders[path], field)), units)
    return datasets
