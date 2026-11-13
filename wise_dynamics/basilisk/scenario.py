"""The Basilisk plumbing every scenario runs on: one simulation, run once and recorded."""

from collections.abc import Mapping
from typing import Any

import numpy as np
from Basilisk.simulation import svIntegrators
from Basilisk.utilities import SimulationBaseClass, macros

__all__ = ["Scenario"]

_PROCESS = "scenario_process"
_GROUND_PROCESS = "ground_process"
_GROUND_TASK = "ground_task"
_PROCESS_PRIORITY = 2
_GROUND_PROCESS_PRIORITY = 1
_DYNAMICS_TASK = "dynamics_task"
_SENSOR_TASK = "sensor_task"
_FSW_TASK = "fsw_task"
_RECORD_TASK = "record_task"
_DYNAMICS_PRIORITY = 3
_SENSOR_PRIORITY = 2
_FSW_PRIORITY = 1
_RECORD_PRIORITY = 0
_MAX_STEP_NS = 1_000_000_000
_MIN_STEP_NS = 1_000_000


def _ralston_rk3(spacecraft: Any) -> Any:
    """Ralston's third-order Runge-Kutta.

    Args:
        spacecraft: The Basilisk dynamic object it integrates.

    Returns:
        The integrator.
    """
    return svIntegrators.svIntegratorRungeKutta3(
        spacecraft,
        [[0.0, 0.0, 0.0], [1 / 2, 0.0, 0.0], [0.0, 3 / 4, 0.0]],
        [2 / 9, 1 / 3, 4 / 9],
        [0.0, 1 / 2, 3 / 4],
    )


def _rkf78(spacecraft: Any) -> Any:
    """Runge-Kutta-Fehlberg 7(8) at the tolerances it sets.

    Args:
        spacecraft: The Basilisk dynamic object it integrates.

    Returns:
        The integrator.
    """
    integrator = svIntegrators.svIntegratorRKF78(spacecraft)
    integrator.setRelativeTolerance(1e-10)
    integrator.setAbsoluteTolerance(1e-12)
    return integrator


_FIDELITY = {"hi": _rkf78, "med": svIntegrators.svIntegratorRK4, "lo": _ralston_rk3}
"""Fidelity to the factory building its integrator for a spacecraft."""


def _step_ns(fsw_ns: int) -> int:
    """The largest step of at most 1 s that divides the flight software's period, in nanoseconds.

    Args:
        fsw_ns: The flight software's period, in nanoseconds.

    Returns:
        The step.

    Raises:
        ValueError: No step of at least 1 ms divides the period.
    """
    fewest = -(-fsw_ns // _MAX_STEP_NS)
    counts = (n for n in range(fewest, fsw_ns // _MIN_STEP_NS + 1) if fsw_ns % n == 0)
    count = next(counts, None)
    if count is None:
        raise ValueError(f"no step of 1 ms to 1 s divides the {fsw_ns} ns flight-software period")
    return fsw_ns // count


class Scenario:
    """Builds a Basilisk simulation from what a subclass sets up, runs it once and records it.

    A subclass's constructor calls `super().__init__()`, builds its vehicles and places them, and
    sets `vehicles`, the vehicles `run` adds to the simulation in order, and `record_s`, the
    recording period in seconds. `run` reads them, and `fidelity`.

    Attributes:
        fidelity: `"hi"`, `"med"` or `"lo"`, the integrator `run` gives every vehicle's spacecraft,
            as the README's Scenarios section defines them.
    """

    fidelity: str = "lo"

    def __init__(self) -> None:
        """Create the empty Basilisk simulation `run` builds into."""
        self._integrators: list[Any] = []
        self._simulation = SimulationBaseClass.SimBaseClass()
        self._executed = False

    def run(
        self,
        fsw_module: Any,
        fsw_fields: Mapping[str, tuple[Any, str, str]],
        *,
        duration_s: float,
        ground: Any = None,
    ) -> dict[str, tuple[np.ndarray, str]]:
        """Run the vehicles with one flight-software module and record them all.

        Gives every vehicle's spacecraft the integrator of `fidelity`, then builds the simulation
        with one process holding four tasks, which run in this order at each instant they share. The
        dynamics task holds each vehicle's `models()` and steps at the largest step of at most 1 s
        that divides `fsw_module.period_s`. The sensor task, every `fsw_module.period_s`, holds each
        vehicle's `sensor_models` for that period, which so sample the state of that instant. The
        flight-software task, every `fsw_module.period_s`, holds `fsw_module`, which so reads the
        sensors' output of that instant. The record task, every `record_s` from t = 0, holds one
        Basilisk recorder per entry of each vehicle's `recorded_fields()` and of `fsw_fields`, so
        truth and flight output are recorded from the same instant. Runs from t = 0 for
        `duration_s`.

        Args:
            fsw_module: Any Basilisk module declaring `period_s`, its period in seconds, for
                example one reading a vehicle's `readers()`.
            fsw_fields: Run-file dataset path to the `(message, field, units)` it records from the
                flight software, in the shape of a vehicle's `recorded_fields()`; may be empty.
            duration_s: Simulated time to run, in seconds.
            ground: Optionally, any Basilisk module declaring `period_s`, its period in seconds,
                run from t = 0 in `ground_process`.

        Returns:
            Dataset path to `(array, units)`: `t_s`, the sample times in seconds as shape `(N,)`,
            then one entry per vehicle field and per `fsw_fields` entry, each of `N` samples.

        Raises:
            RuntimeError: The scenario has already run.
            ValueError: No step of at least 1 ms divides `fsw_module.period_s`, or `record_s` is
                not a whole number of steps; raised before anything runs.
        """
        if self._executed:
            raise RuntimeError("this scenario has already run; build a new one for another run")
        fsw_ns = macros.sec2nano(fsw_module.period_s)
        record_ns = macros.sec2nano(self.record_s)
        step_ns = _step_ns(fsw_ns)
        if record_ns % step_ns:
            raise ValueError(
                f"record_s {self.record_s} s is not a whole number of the {step_ns} ns step that "
                f"period_s {fsw_module.period_s} s gives"
            )
        self._executed = True

        make_integrator = _FIDELITY[self.fidelity]
        fields = {}
        for vehicle in self.vehicles:
            integrator = make_integrator(vehicle.spacecraft)
            vehicle.spacecraft.setIntegrator(integrator)
            self._integrators.append(integrator)
            fields.update(vehicle.recorded_fields())
        fields.update(fsw_fields)
        recorders = {path: message.recorder(record_ns) for path, (message, _, _) in fields.items()}

        tasks = (
            (_DYNAMICS_TASK, step_ns, _DYNAMICS_PRIORITY),
            (_SENSOR_TASK, fsw_ns, _SENSOR_PRIORITY),
            (_FSW_TASK, fsw_ns, _FSW_PRIORITY),
            (_RECORD_TASK, record_ns, _RECORD_PRIORITY),
        )
        process = self._simulation.CreateNewProcess(_PROCESS, _PROCESS_PRIORITY)
        for name, task_ns, priority in tasks:
            process.addTask(self._simulation.CreateNewTask(name, task_ns), priority)
        for vehicle in self.vehicles:
            for model in vehicle.models():
                self._simulation.AddModelToTask(_DYNAMICS_TASK, model)
            for sensor in vehicle.sensor_models(fsw_module.period_s):
                self._simulation.AddModelToTask(_SENSOR_TASK, sensor)
        self._simulation.AddModelToTask(_FSW_TASK, fsw_module)
        if ground is not None:
            ground_process = self._simulation.CreateNewProcess(
                _GROUND_PROCESS, _GROUND_PROCESS_PRIORITY
            )
            ground_task = self._simulation.CreateNewTask(
                _GROUND_TASK, macros.sec2nano(ground.period_s)
            )
            ground_process.addTask(ground_task)
            self._simulation.AddModelToTask(_GROUND_TASK, ground)
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
