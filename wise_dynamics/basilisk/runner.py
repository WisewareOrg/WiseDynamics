"""Runs one case: a plant, one flight-software module and recorders, returned as a `Recording`."""

from collections.abc import Mapping
from typing import Any

from Basilisk.utilities import macros

from wise_dynamics.basilisk.bsk_types import Message, SysModel
from wise_dynamics.basilisk.plant import Plant
from wise_dynamics.basilisk.recording import Recording

__all__ = ["run_case"]


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
