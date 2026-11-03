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
    """A prebuilt operating environment, the vehicle placed in it, and the Basilisk simulation
    that runs them once.

    A subclass defines its environment, its placement, its recording period and its vehicle class
    as class attributes and takes no constructor arguments. A scenario builds its vehicle and holds
    it in `vehicles`; `disperse` draws the scenario's dispersions, overrides the placement of one
    instance and builds its vehicle again from the drawn values. `run` builds the simulation: it
    checks the placement through `initial_state` before anything is changed, adds each vehicle's
    models, and has `_place` attach the gravity bodies and write the initial state onto the
    vehicle's `spacecraft`.

    Attributes:
        mu_m3_s2: Gravity body name to its gravitational parameter, in m^3/s^2.
        standard_dispersion: Placement attribute to its per-axis 1-sigma, drawn in this order by
            `disperse`.
        step_s: The integration and sensor step, in seconds.
        record_s: The recording period, in seconds.
        vehicle_type: The class of the vehicle the scenario builds.
        vehicle: The vehicle the scenario built; `disperse` replaces it.
        vehicles: The vehicles in the simulation, in order.
        draws: What `disperse` drew and applied, name to value; empty before `disperse`.
    """

    mu_m3_s2: ClassVar[Mapping[str, float]]
    standard_dispersion: ClassVar[Mapping[str, float]]
    step_s: float
    record_s: float
    vehicle_type: ClassVar[type[Vehicle]]

    def __init__(self) -> None:
        self.vehicles: list[Vehicle] = []
        self.draws: dict[str, np.ndarray] = {}
        self._simulation = SimulationBaseClass.SimBaseClass()
        self._executed = False
        self.vehicle = self.vehicle_type()
        self.vehicles.append(self.vehicle)

    @classmethod
    def _draw(cls, rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Overrides drawn from `rng`: the scenario's `standard_dispersion` entries, then its
        vehicle class's, each drawn as `Vehicle.draw` draws."""
        return {**Vehicle.draw.__func__(cls, rng), **cls.vehicle_type.draw(rng)}

    def disperse(self, rng: np.random.Generator, *, noise_seed: int) -> None:
        """Draw this scenario's dispersions from `rng`, the placement's then the vehicle class's,
        set each drawn placement value on this instance, and build the vehicle again with the drawn
        vehicle values and `noise_seed` as constructor keywords. The new vehicle replaces
        `vehicle`, so readers and other handles on it are taken after dispersing. `draws` records
        what was applied.

        Args:
            rng: The generator the dispersions are drawn from.
            noise_seed: The new vehicle's sensor-noise seed.

        Raises:
            RuntimeError: The scenario has already run.
        """
        if self._executed:
            raise RuntimeError("this scenario has already run; build a new one for another run")
        draws = self._draw(rng)
        placement = set(self.standard_dispersion)
        vehicle_draws = {name: value for name, value in draws.items() if name not in placement}
        vehicle = self.vehicle_type(**vehicle_draws, noise_seed=noise_seed)
        for name in placement:
            setattr(self, name, draws[name])
        self.vehicles[self.vehicles.index(self.vehicle)] = vehicle
        self.vehicle = vehicle
        self.draws = draws

    @abstractmethod
    def initial_state(self) -> dict[str, np.ndarray]:
        """The initial condition the scenario's placement gives its vehicle, by key, valid before
        `run`. A subclass documents its keys.

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
        fsw_fields: Mapping[str, tuple[Any, str, str]],
        *,
        duration_s: float,
    ) -> dict[str, tuple[np.ndarray, str]]:
        """Run the vehicles with one flight-software module and record them all.

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

        Returns:
            Dataset path to `(array, units)`: `t_s`, the sample times in seconds as shape `(N,)`,
            then one entry per vehicle field and per `fsw_fields` entry, each of `N` samples.

        Raises:
            ValueError: `step_s`, `record_s` or `duration_s` is not positive, an `fsw_fields` path
                is `t_s` or a vehicle dataset path, a recorded field is absent from its message, or
                the placement is invalid.
            RuntimeError: The scenario has already run.
        """
        if self._executed:
            raise RuntimeError("this scenario has already run; build a new one for another run")
        if not self.step_s > 0:
            raise ValueError(f"step must be positive, got {self.step_s} s")
        if not self.record_s > 0:
            raise ValueError(f"recording period must be positive, got {self.record_s} s")
        if not duration_s > 0:
            raise ValueError(f"duration must be positive, got {duration_s} s")
        vehicle_fields = {}
        for vehicle in self.vehicles:
            vehicle_fields.update(vehicle.recorded_fields())
        reused = sorted(set(fsw_fields) & {"t_s", *vehicle_fields})
        if reused:
            raise ValueError(f"fsw_fields reuses the run's own dataset paths {reused}")

        fields = {**vehicle_fields, **fsw_fields}
        period_ns = macros.sec2nano(self.record_s)
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
