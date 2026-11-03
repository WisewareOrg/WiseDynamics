"""The interface every Basilisk scenario meets: a vehicle placed in an environment, run once."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["Scenario"]

_PROCESS = "scenario_process"
_TASK = "scenario_task"


class Scenario(ABC):
    """An operating environment, a vehicle placed in it, and the Basilisk simulation that runs
    them once.

    A subclass takes its environment and placement as keywords defaulting to class attributes of
    the same name, checks them, calls `super().__init__(vehicle, step_s=...)`, then attaches its
    gravity bodies to `vehicle.spacecraft` and sets its initial state.

    Attributes:
        mu_m3_s2: Gravity body name to its gravitational parameter, in m^3/s^2.
        standard_dispersion: Placement keyword to its per-axis 1-sigma, drawn in this order by
            `draw`.
        vehicle: The vehicle the scenario was given.
    """

    mu_m3_s2: ClassVar[Mapping[str, float]]
    standard_dispersion: ClassVar[Mapping[str, float]]

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
        process = self._simulation.CreateNewProcess(_PROCESS)
        process.addTask(self._simulation.CreateNewTask(_TASK, macros.sec2nano(step_s)))
        for model in vehicle.models():
            self._simulation.AddModelToTask(_TASK, model)
        self._executed = False

    @classmethod
    def draw(cls, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Keyword overrides drawn from `rng`: each `standard_dispersion` entry, in order, as the
        keyword's class default plus independent normal samples of that 1-sigma."""
        draws = {}
        for name, std in cls.standard_dispersion.items():
            nominal = np.asarray(getattr(cls, name), dtype=float)
            draws[name] = nominal + rng.normal(0.0, std, size=nominal.shape)
        return draws

    @abstractmethod
    def initial_state(self) -> dict[str, np.ndarray]:
        """The vehicle's initial condition as the simulation holds it, by key. A subclass
        documents its keys."""

    def run(
        self,
        fsw_module: Any,
        *,
        fsw_fields: Mapping[str, tuple[Any, str, str]],
        duration_s: float,
        record_s: float = 10.0,
    ) -> dict[str, tuple[np.ndarray, str]]:
        """Run the vehicle with one flight-software module and record both.

        Adds `fsw_module` to the task after the vehicle's models, attaches one Basilisk recorder
        per entry of `vehicle.recorded_fields()` and of `fsw_fields`, each sampling every
        `record_s` from t = 0, and runs from t = 0 for `duration_s`. The arguments are checked
        before the simulation is changed.

        Args:
            fsw_module: Any Basilisk module, for example one reading `vehicle.readers()`.
            fsw_fields: Run-file dataset path to the `(message, field, units)` it records from the
                flight software, in the shape of `vehicle.recorded_fields()`; may be empty.
            duration_s: Simulated time to run, in seconds.
            record_s: Recording period, in seconds.

        Returns:
            Dataset path to `(array, units)`: `t_s`, the sample times in seconds as shape `(N,)`,
            then one entry per vehicle field and per `fsw_fields` entry, each of `N` samples.

        Raises:
            ValueError: `record_s` or `duration_s` is not positive, an `fsw_fields` path is `t_s`
                or a vehicle dataset path, or a recorded field is absent from its message.
            RuntimeError: The scenario has already run.
        """
        if self._executed:
            raise RuntimeError("this scenario has already run; build a new one for another run")
        if not record_s > 0:
            raise ValueError(f"recording period must be positive, got {record_s} s")
        if not duration_s > 0:
            raise ValueError(f"duration must be positive, got {duration_s} s")
        vehicle_fields = self.vehicle.recorded_fields()
        reused = sorted(set(fsw_fields) & {"t_s", *vehicle_fields})
        if reused:
            raise ValueError(f"fsw_fields reuses the run's own dataset paths {reused}")

        fields = {**vehicle_fields, **fsw_fields}
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
        datasets = {"t_s": (np.asarray(first.times()) / macros.sec2nano(1.0), "s")}
        for path, (_, field, units) in fields.items():
            datasets[path] = (np.array(getattr(recorders[path], field)), units)
        return datasets
