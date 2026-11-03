"""The interface every Basilisk scenario meets: vehicles placed in an environment, run once."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["Scenario"]

_PROCESS = "scenario_process"
_DYNAMICS_TASK = "dynamics_task"
_FSW_TASK = "fsw_task"
_DYNAMICS_PRIORITY = 2
_FSW_PRIORITY = 1


class Scenario(ABC):
    """A prebuilt operating environment, the vehicles placed in it, and the Basilisk simulation
    that runs them once.

    A subclass defines its environment and placement as class attributes and takes no
    constructor arguments. `disperse` overrides the placement of one instance, and `add` places a
    vehicle. `run` builds the simulation: it checks the placement through `initial_state` before
    anything is changed, adds each vehicle's models, and has `_place` attach the gravity bodies and
    write the initial state onto the vehicle's `spacecraft`.

    The simulation runs two tasks. The dynamics task, every `step_s`, holds the vehicles' models
    and sensors. The flight-software task, every `fsw_step_s`, holds the flight-software module and
    the recorders, and runs after the dynamics task at each instant they share, so the flight
    software reads the sensors' output of that instant.

    Attributes:
        mu_m3_s2: Gravity body name to its gravitational parameter, in m^3/s^2.
        standard_dispersion: Placement attribute to its per-axis 1-sigma, drawn in this order by
            `draw`.
        step_s: The dynamics task's period, the integration and sensor step, in seconds.
        fsw_step_s: The flight-software task's period, in seconds, a whole multiple of `step_s`.
        vehicles: The vehicles added, in order.
    """

    mu_m3_s2: ClassVar[Mapping[str, float]]
    standard_dispersion: ClassVar[Mapping[str, float]]
    step_s: float
    fsw_step_s: float

    def __init__(self) -> None:
        self.vehicles: list[Vehicle] = []
        self._simulation = SimulationBaseClass.SimBaseClass()
        self._executed = False

    draw = classmethod(Vehicle.draw.__func__)
    """Placement overrides drawn from `rng`, as `Vehicle.draw` draws a vehicle's."""

    def disperse(self, draws: Mapping[str, Any]) -> None:
        """Set each entry of `draws` as this instance's value of that placement attribute, as
        `draw` returns them.

        Raises:
            ValueError: An entry names no `standard_dispersion` attribute.
        """
        unknown = sorted(set(draws) - set(self.standard_dispersion))
        if unknown:
            raise ValueError(f"{type(self).__name__} does not disperse {unknown}")
        for name, value in draws.items():
            setattr(self, name, value)

    def add(self, vehicle: Vehicle) -> None:
        """Place `vehicle` at the scenario's placement; `run` adds it to the simulation.

        Raises:
            RuntimeError: The scenario has already run.
            ValueError: A vehicle is already added; a scenario takes one.
        """
        if self._executed:
            raise RuntimeError("this scenario has already run; build a new one for another run")
        if self.vehicles:
            raise ValueError(f"{type(self).__name__} takes one vehicle and already holds it")
        self.vehicles.append(vehicle)

    @abstractmethod
    def initial_state(self) -> dict[str, np.ndarray]:
        """The initial condition the scenario's placement gives its vehicle, by key, valid before
        `add` and `run`. A subclass documents its keys.

        Raises:
            ValueError: The placement is invalid.
        """

    @abstractmethod
    def _place(self, vehicle: Vehicle) -> None:
        """Attach the gravity bodies to `vehicle.spacecraft` and write its initial state. `run`
        calls it once per vehicle, after `initial_state` has checked the placement."""

    def run(
        self,
        fsw_module: Any,
        *,
        fsw_fields: Mapping[str, tuple[Any, str, str]],
        duration_s: float,
        record_s: float = 10.0,
    ) -> dict[str, tuple[np.ndarray, str]]:
        """Run the vehicles with one flight-software module and record them all.

        Builds the simulation with one process holding the dynamics task at `step_s` and the
        flight-software task at `fsw_step_s`. Adds each vehicle's models to the dynamics task and
        places the vehicle, then adds `fsw_module` to the flight-software task, followed by one
        Basilisk recorder per entry of each vehicle's `recorded_fields()` and of `fsw_fields`, each
        sampling every `record_s` from t = 0. Runs from t = 0 for `duration_s`. The arguments and
        the placement are checked before anything is changed.

        Args:
            fsw_module: Any Basilisk module, for example one reading a vehicle's `readers()`.
            fsw_fields: Run-file dataset path to the `(message, field, units)` it records from the
                flight software, in the shape of a vehicle's `recorded_fields()`; may be empty.
            duration_s: Simulated time to run, in seconds.
            record_s: Recording period, in seconds, a whole multiple of `fsw_step_s`.

        Returns:
            Dataset path to `(array, units)`: `t_s`, the sample times in seconds as shape `(N,)`,
            then one entry per vehicle field and per `fsw_fields` entry, each of `N` samples.

        Raises:
            ValueError: No vehicle is added, `step_s`, `fsw_step_s`, `record_s` or `duration_s` is
                not positive, `fsw_step_s` is not a whole multiple of `step_s` or `record_s` of
                `fsw_step_s`, an `fsw_fields` path is `t_s` or a vehicle dataset path, a recorded
                field is absent from its message, or the placement is invalid.
            RuntimeError: The scenario has already run.
        """
        if self._executed:
            raise RuntimeError("this scenario has already run; build a new one for another run")
        if not self.vehicles:
            raise ValueError("add a vehicle before running the scenario")
        if not self.step_s > 0:
            raise ValueError(f"step must be positive, got {self.step_s} s")
        if not self.fsw_step_s > 0:
            raise ValueError(f"flight-software step must be positive, got {self.fsw_step_s} s")
        if not record_s > 0:
            raise ValueError(f"recording period must be positive, got {record_s} s")
        if not duration_s > 0:
            raise ValueError(f"duration must be positive, got {duration_s} s")
        step_ns = macros.sec2nano(self.step_s)
        fsw_step_ns = macros.sec2nano(self.fsw_step_s)
        period_ns = macros.sec2nano(record_s)
        if fsw_step_ns % step_ns:
            raise ValueError(
                f"flight-software step {self.fsw_step_s} s is not a whole multiple of the step "
                f"{self.step_s} s"
            )
        if period_ns % fsw_step_ns:
            raise ValueError(
                f"recording period {record_s} s is not a whole multiple of the flight-software "
                f"step {self.fsw_step_s} s"
            )
        vehicle_fields = {}
        for vehicle in self.vehicles:
            vehicle_fields.update(vehicle.recorded_fields())
        reused = sorted(set(fsw_fields) & {"t_s", *vehicle_fields})
        if reused:
            raise ValueError(f"fsw_fields reuses the run's own dataset paths {reused}")

        fields = {**vehicle_fields, **fsw_fields}
        recorders = {path: message.recorder(period_ns) for path, (message, _, _) in fields.items()}
        for path, (_, field, _) in fields.items():
            if not isinstance(getattr(type(recorders[path]), field, None), property):
                raise ValueError(f"dataset {path!r}: its message has no field {field!r}")

        self.initial_state()
        self._executed = True
        process = self._simulation.CreateNewProcess(_PROCESS)
        process.addTask(self._simulation.CreateNewTask(_DYNAMICS_TASK, step_ns), _DYNAMICS_PRIORITY)
        process.addTask(self._simulation.CreateNewTask(_FSW_TASK, fsw_step_ns), _FSW_PRIORITY)
        for vehicle in self.vehicles:
            for model in vehicle.models():
                self._simulation.AddModelToTask(_DYNAMICS_TASK, model)
            self._place(vehicle)
        self._simulation.AddModelToTask(_FSW_TASK, fsw_module)
        for recorder in recorders.values():
            self._simulation.AddModelToTask(_FSW_TASK, recorder)
        self._simulation.InitializeSimulation()
        self._simulation.ConfigureStopTime(macros.sec2nano(duration_s))
        self._simulation.ExecuteSimulation()

        first = next(iter(recorders.values()))
        datasets = {"t_s": (np.asarray(first.times()) / macros.sec2nano(1.0), "s")}
        for path, (_, field, units) in fields.items():
            datasets[path] = (np.array(getattr(recorders[path], field)), units)
        return datasets
