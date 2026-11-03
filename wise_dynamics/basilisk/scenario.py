"""The Basilisk plumbing every scenario runs on: one simulation, run once and recorded."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

__all__ = ["Scenario"]

_PROCESS = "scenario_process"
_TASK = "scenario_task"


class Scenario:
    """Builds a Basilisk simulation from what a subclass sets up, runs it once and records it.

    A subclass's constructor calls `super().__init__()`, builds its vehicles and places them, and
    sets `vehicles`, `step_s` and `record_s`; `run` reads them.

    Attributes:
        vehicles: The vehicles `run` adds to the simulation, in order.
        step_s: The integration and sensor step, in seconds.
        record_s: The recording period, in seconds.
    """

    def __init__(self) -> None:
        self._simulation = SimulationBaseClass.SimBaseClass()
        self._executed = False

    def run(
        self,
        fsw_module: Any,
        fsw_fields: Mapping[str, tuple[Any, str, str]],
        *,
        duration_s: float,
    ) -> dict[str, tuple[np.ndarray, str]]:
        """Run the vehicles with one flight-software module and record them all.

        Builds the simulation with one process holding one task at `step_s`, adds each vehicle's
        models, adds `fsw_module` after them, attaches one Basilisk recorder per entry of each
        vehicle's `recorded_fields()` and of `fsw_fields`, each sampling every `record_s` from
        t = 0, and runs from t = 0 for `duration_s`.

        Args:
            fsw_module: Any Basilisk module, for example one reading a vehicle's `readers()`.
            fsw_fields: Run-file dataset path to the `(message, field, units)` it records from the
                flight software, in the shape of a vehicle's `recorded_fields()`; may be empty.
            duration_s: Simulated time to run, in seconds.

        Returns:
            Dataset path to `(array, units)`: `t_s`, the sample times in seconds as shape `(N,)`,
            then one entry per vehicle field and per `fsw_fields` entry, each of `N` samples.

        Raises:
            RuntimeError: The scenario has already run.
        """
        if self._executed:
            raise RuntimeError("this scenario has already run; build a new one for another run")
        self._executed = True

        fields = {}
        for vehicle in self.vehicles:
            fields.update(vehicle.recorded_fields())
        fields.update(fsw_fields)
        period_ns = macros.sec2nano(self.record_s)
        recorders = {path: message.recorder(period_ns) for path, (message, _, _) in fields.items()}

        process = self._simulation.CreateNewProcess(_PROCESS)
        process.addTask(self._simulation.CreateNewTask(_TASK, macros.sec2nano(self.step_s)))
        for vehicle in self.vehicles:
            for model in vehicle.models():
                self._simulation.AddModelToTask(_TASK, model)
        self._simulation.AddModelToTask(_TASK, fsw_module)
        for recorder in recorders.values():
            self._simulation.AddModelToTask(_TASK, recorder)
        self._simulation.InitializeSimulation()
        self._simulation.ConfigureStopTime(macros.sec2nano(duration_s))
        self._simulation.ExecuteSimulation()

        first = next(iter(recorders.values()))
        datasets = {"t_s": (np.asarray(first.times()) / macros.sec2nano(1.0), "s")}
        for path, (_, field, units) in fields.items():
            datasets[path] = (np.array(getattr(recorders[path], field)), units)
        return datasets
