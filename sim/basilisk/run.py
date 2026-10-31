"""A one-case runner: adds a flight-software module to a plant's task, records the plant's
truth messages and the module's own output messages, and runs."""

from typing import Any, Protocol

from Basilisk.utilities import macros


class Plant(Protocol):
    """The interface `run_case` needs from a plant: any class with these attributes and
    methods works, not only `ImuSpacecraftModel`."""

    sim: Any
    task_name: str

    def readers(self) -> dict[str, Any]: ...
    def truth_messages(self) -> dict[str, Any]: ...


def run_case(
    plant: Plant,
    fsw_module: Any,
    fsw_outputs: dict[str, Any],
    duration_s: float,
    record_s: float = 10.0,
) -> dict[str, Any]:
    """Adds `fsw_module` (any Basilisk `SysModel`) to `plant`'s task, records `plant`'s own
    truth messages (`plant.truth_messages()`, a `{role: msg}` dict) and `fsw_outputs` (the
    caller's own `{role: msg}` dict for `fsw_module`'s output messages) at `record_s`, then runs
    for `duration_s`. Returns the live recorders, keyed by each message's role; each recorder
    exposes its recorded fields as attributes and its sample times via `.times()`."""
    plant.sim.AddModelToTask(plant.task_name, fsw_module)

    sampling_time = macros.sec2nano(record_s)
    recorders: dict[str, Any] = {}
    for role, msg in plant.truth_messages().items():
        recorders[role] = msg.recorder(sampling_time)
    for role, msg in fsw_outputs.items():
        recorders[role] = msg.recorder(sampling_time)
    for recorder in recorders.values():
        plant.sim.AddModelToTask(plant.task_name, recorder)

    plant.sim.InitializeSimulation()
    plant.sim.ConfigureStopTime(macros.sec2nano(duration_s))
    plant.sim.ExecuteSimulation()

    return recorders
