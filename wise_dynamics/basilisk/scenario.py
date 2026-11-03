"""The Basilisk plumbing every scenario runs on: one simulation, run once and recorded."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from Basilisk.simulation import svIntegrators
from Basilisk.utilities import SimulationBaseClass, macros

__all__ = ["Scenario"]

_PROCESS = "scenario_process"
_DYNAMICS_TASK = "dynamics_task"
_FSW_TASK = "fsw_task"
_RECORD_TASK = "record_task"
_DYNAMICS_PRIORITY = 2
_FSW_PRIORITY = 1
_RECORD_PRIORITY = 0
_FIDELITY = {
    "hi": (0.02, svIntegrators.svIntegratorRK4),
    "med": (0.05, svIntegrators.svIntegratorRK4),
    "lo": (0.05, svIntegrators.svIntegratorRK2),
}


class Scenario:
    """Builds a Basilisk simulation from what a subclass sets up, runs it once and records it.

    A subclass's constructor calls `super().__init__(fidelity)`, builds its vehicles and places
    them, and sets `vehicles` and `record_s`; `run` reads them.

    The fidelity sets the integration step and the integrator `run` gives every vehicle's
    spacecraft: `"hi"` is RK4 at 0.02 s, `"med"` RK4 at 0.05 s and `"lo"` RK2 at 0.05 s.

    Args:
        fidelity: `"hi"`, `"med"` or `"lo"`.

    Attributes:
        fidelity: The fidelity the scenario was built at.
        step_s: The integration and sensor step, in seconds, set by the fidelity.
        vehicles: The vehicles `run` adds to the simulation, in order.
        record_s: The recording period, in seconds.
    """

    def __init__(self, fidelity: str = "lo") -> None:
        self.fidelity = fidelity
        self.step_s, self._integrator_type = _FIDELITY[fidelity]
        self._integrators: list[Any] = []
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

        Gives every vehicle's spacecraft the fidelity's integrator, then builds the simulation with
        one process holding three tasks, which run in this order at
        each instant they share. The dynamics task, every `step_s`, holds each vehicle's models.
        The flight-software task, every `fsw_module.period_s`, holds `fsw_module`, which so reads
        the sensors' output of that instant. The record task, every `record_s` from t = 0, holds
        one Basilisk recorder per entry of each vehicle's `recorded_fields()` and of `fsw_fields`,
        so truth and flight output are recorded from the same instant. Runs from t = 0 for
        `duration_s`.

        Args:
            fsw_module: Any Basilisk module declaring `period_s`, its period in seconds, for
                example one reading a vehicle's `readers()`.
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
            integrator = self._integrator_type(vehicle.spacecraft)
            vehicle.spacecraft.setIntegrator(integrator)
            self._integrators.append(integrator)
            fields.update(vehicle.recorded_fields())
        fields.update(fsw_fields)
        period_ns = macros.sec2nano(self.record_s)
        recorders = {path: message.recorder(period_ns) for path, (message, _, _) in fields.items()}

        process = self._simulation.CreateNewProcess(_PROCESS)
        process.addTask(
            self._simulation.CreateNewTask(_DYNAMICS_TASK, macros.sec2nano(self.step_s)),
            _DYNAMICS_PRIORITY,
        )
        process.addTask(
            self._simulation.CreateNewTask(_FSW_TASK, macros.sec2nano(fsw_module.period_s)),
            _FSW_PRIORITY,
        )
        process.addTask(self._simulation.CreateNewTask(_RECORD_TASK, period_ns), _RECORD_PRIORITY)
        for vehicle in self.vehicles:
            for model in vehicle.models():
                self._simulation.AddModelToTask(_DYNAMICS_TASK, model)
        self._simulation.AddModelToTask(_FSW_TASK, fsw_module)
        for recorder in recorders.values():
            self._simulation.AddModelToTask(_RECORD_TASK, recorder)
        self._simulation.InitializeSimulation()
        self._simulation.ConfigureStopTime(macros.sec2nano(duration_s))
        self._simulation.ExecuteSimulation()

        first = next(iter(recorders.values()))
        datasets = {"t_s": (np.asarray(first.times()) / macros.sec2nano(1.0), "s")}
        for path, (_, field, units) in fields.items():
            datasets[path] = (np.array(getattr(recorders[path], field)), units)
        return datasets
