"""The interface every Basilisk scenario meets: a vehicle placed in an environment, run once."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["Scenario"]

_TASK = "scenario_task"
_NANOSECONDS_PER_SECOND = 1e9


class Scenario(ABC):
    """An operating environment, a vehicle placed in it, and the Basilisk simulation that runs
    them once.

    A subclass takes its environment and placement as keywords defaulting to class attributes of
    the same name, checks them, calls `super().__init__(vehicle, step_s=...)`, then attaches its
    gravity bodies to `vehicle.spacecraft` and sets its initial state.

    Attributes:
        mu: Gravity body name to its gravitational parameter, in m^3/s^2.
        vehicle: The vehicle the scenario was given.
    """

    mu: ClassVar[Mapping[str, float]]

    def __init__(self, vehicle: Vehicle, *, step_s: float) -> None:
        """Create the simulation, with one process holding one task, and add the vehicle's models
        to the task.

        Args:
            vehicle: The vehicle to place.
            step_s: The task's period, in seconds: the integration and sensor step.

        Raises:
            ValueError: `step_s` is not positive.
        """
        if not step_s > 0:
            raise ValueError(f"step must be positive, got {step_s} s")
        self.vehicle = vehicle
        self.step_s = step_s
        self._simulation = SimulationBaseClass.SimBaseClass()
        process = self._simulation.CreateNewProcess("scenario_process")
        process.addTask(self._simulation.CreateNewTask(_TASK, macros.sec2nano(step_s)))
        for model in vehicle.models():
            self._simulation.AddModelToTask(_TASK, model)
        self._executed = False

    @abstractmethod
    def initial_state(self) -> dict[str, np.ndarray]:
        """The vehicle's initial condition as the simulation holds it, by key. A subclass
        documents its keys."""

    def run(
        self, fsw_module: Any, *, duration_s: float, record_s: float = 10.0
    ) -> dict[str, tuple[np.ndarray, str]]:
        """Run the vehicle with one flight-software module and record the vehicle's fields.

        Adds `fsw_module` to the task after the vehicle's models, attaches one Basilisk recorder
        per entry of `vehicle.recorded_fields()` sampling every `record_s` from t = 0, and runs
        from t = 0 for `duration_s`. The arguments are checked before the simulation is changed.

        Args:
            fsw_module: Any Basilisk module, for example one reading `vehicle.readers()`.
            duration_s: Simulated time to run, in seconds.
            record_s: Recording period, in seconds.

        Returns:
            Dataset path to `(array, units)`: `t_s`, the sample times in seconds as shape `(N,)`,
            then one entry per recorded field, each of `N` samples.

        Raises:
            ValueError: `record_s` or `duration_s` is not positive, or a recorded field is absent
                from its message.
            RuntimeError: The scenario has already run.
        """
        if self._executed:
            raise RuntimeError("this scenario has already run; build a new one for another run")
        if not record_s > 0:
            raise ValueError(f"recording period must be positive, got {record_s} s")
        if not duration_s > 0:
            raise ValueError(f"duration must be positive, got {duration_s} s")

        fields = self.vehicle.recorded_fields()
        period_ns = macros.sec2nano(record_s)
        recorders = {path: message.recorder(period_ns) for path, (message, _, _) in fields.items()}
        for path, (_, field, _) in fields.items():
            if not isinstance(getattr(type(recorders[path]), field, None), property):
                raise ValueError(f"dataset {path!r}: its message has no field {field!r}")

        self._executed = True
        self._simulation.AddModelToTask(_TASK, fsw_module)
        for recorder in recorders.values():
            self._simulation.AddModelToTask(_TASK, recorder)
        self._simulation.InitializeSimulation()
        self._simulation.ConfigureStopTime(macros.sec2nano(duration_s))
        self._simulation.ExecuteSimulation()

        first = next(iter(recorders.values()))
        datasets = {"t_s": (np.asarray(first.times()) / _NANOSECONDS_PER_SECOND, "s")}
        for path, (_, field, units) in fields.items():
            datasets[path] = (np.array(getattr(recorders[path], field)), units)
        return datasets
