"""The interface every Basilisk scenario meets: vehicles placed in an environment, run once."""

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
    """A prebuilt operating environment, the vehicles placed in it, and the Basilisk simulation
    that runs them once.

    A subclass defines its environment and placement as class attributes and takes no
    constructor arguments. `disperse` overrides the placement of one instance, and `add` places a
    vehicle. `run` builds the simulation: it checks the placement through `initial_state` before
    anything is changed, adds each vehicle's models, and has `_place` attach the gravity bodies and
    write the initial state onto the vehicle's `spacecraft`.

    Attributes:
        mu_m3_s2: Gravity body name to its gravitational parameter, in m^3/s^2.
        standard_dispersion: Placement attribute to its per-axis 1-sigma, drawn in this order by
            `draw`.
        step_s: The integration and sensor step, in seconds.
        vehicles: The vehicles added, in order.
    """

    mu_m3_s2: ClassVar[Mapping[str, float]]
    standard_dispersion: ClassVar[Mapping[str, float]]
    step_s: float

    def __init__(self) -> None:
        self.vehicles: list[Vehicle] = []
        self._simulation = SimulationBaseClass.SimBaseClass()
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
            ValueError: A vehicle is already added; a second one needs a placement of its own.
        """
        if self.vehicles:
            raise ValueError(
                f"{type(self).__name__} already holds a vehicle; a second one needs a placement "
                "of its own"
            )
        self.vehicles.append(vehicle)

    @abstractmethod
    def initial_state(self) -> dict[str, np.ndarray]:
        """The first vehicle's initial condition from the scenario's placement, by key, valid
        before `run`. A subclass documents its keys.

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
        """Run the vehicle with one flight-software module and record both.

        Builds the simulation with one process holding one task at `step_s`, adds each vehicle's
        models and places it, adds `fsw_module` after them, attaches one Basilisk recorder per
        entry of each vehicle's `recorded_fields()` and of `fsw_fields`, each sampling every
        `record_s` from t = 0, and runs from t = 0 for `duration_s`. The arguments and the
        placement are checked before anything is changed.

        Args:
            fsw_module: Any Basilisk module, for example one reading a vehicle's `readers()`.
            fsw_fields: Run-file dataset path to the `(message, field, units)` it records from the
                flight software, in the shape of a vehicle's `recorded_fields()`; may be empty.
            duration_s: Simulated time to run, in seconds.
            record_s: Recording period, in seconds.

        Returns:
            Dataset path to `(array, units)`: `t_s`, the sample times in seconds as shape `(N,)`,
            then one entry per vehicle field and per `fsw_fields` entry, each of `N` samples.

        Raises:
            ValueError: No vehicle is added, `step_s`, `record_s` or `duration_s` is not positive,
                an `fsw_fields` path is `t_s` or a vehicle dataset path, a recorded field is
                absent from its message, or the placement is invalid.
            RuntimeError: The scenario has already run.
        """
        if self._executed:
            raise RuntimeError("this scenario has already run; build a new one for another run")
        if not self.vehicles:
            raise ValueError("add a vehicle before running the scenario")
        if not self.step_s > 0:
            raise ValueError(f"step must be positive, got {self.step_s} s")
        if not record_s > 0:
            raise ValueError(f"recording period must be positive, got {record_s} s")
        if not duration_s > 0:
            raise ValueError(f"duration must be positive, got {duration_s} s")
        vehicle_fields = {}
        for vehicle in self.vehicles:
            vehicle_fields.update(vehicle.recorded_fields())
        reused = sorted(set(fsw_fields) & {"t_s", *vehicle_fields})
        if reused:
            raise ValueError(f"fsw_fields reuses the run's own dataset paths {reused}")

        fields = {**vehicle_fields, **fsw_fields}
        period_ns = macros.sec2nano(record_s)
        recorders = {path: message.recorder(period_ns) for path, (message, _, _) in fields.items()}
        for path, (_, field, _) in fields.items():
            if not isinstance(getattr(type(recorders[path]), field, None), property):
                raise ValueError(f"dataset {path!r}: its message has no field {field!r}")

        self.initial_state()
        self._executed = True
        process = self._simulation.CreateNewProcess(_PROCESS)
        process.addTask(self._simulation.CreateNewTask(_TASK, macros.sec2nano(self.step_s)))
        for vehicle in self.vehicles:
            for model in vehicle.models():
                self._simulation.AddModelToTask(_TASK, model)
            self._place(vehicle)
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
