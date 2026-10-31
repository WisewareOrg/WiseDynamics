"""A one-case runner: adds a flight-software module to a plant's task, records the plant's
truth and IMU messages and the module's own output messages by role, and runs."""

from typing import Any

from Basilisk.utilities import macros

from sim.basilisk.models.imu_spacecraft import ImuSpacecraftModel


def run_case(
    plant: ImuSpacecraftModel,
    fsw_module: Any,
    duration_s: float,
    record_s: float = 10.0,
) -> dict[str, Any]:
    """Adds `fsw_module` (any Basilisk `SysModel`, already wired to `plant`'s messages) to
    `plant`'s task, attaches a recorder at `record_s` to the plant's truth and IMU messages and
    to each of `fsw_module.outputs`' messages (a `{role: (payload_type, msg)}` dict, the
    convention a module built by a host adapter exposes), then runs for `duration_s`. Returns
    the live recorders keyed `"truth"`, `"imu"`, and by each output role name; each recorder
    exposes its recorded fields as attributes and its sample times via `.times()`."""
    plant.sim.AddModelToTask(plant.task_name, fsw_module)

    sampling_time = macros.sec2nano(record_s)
    recorders: dict[str, Any] = {
        "truth": plant.scObject.scStateOutMsg.recorder(sampling_time),
        "imu": plant.imuObject.sensorOutMsg.recorder(sampling_time),
    }
    for role, (_, msg) in fsw_module.outputs.items():
        recorders[role] = msg.recorder(sampling_time)
    for recorder in recorders.values():
        plant.sim.AddModelToTask(plant.task_name, recorder)

    plant.sim.InitializeSimulation()
    plant.sim.ConfigureStopTime(macros.sec2nano(duration_s))
    plant.sim.ExecuteSimulation()

    return recorders
