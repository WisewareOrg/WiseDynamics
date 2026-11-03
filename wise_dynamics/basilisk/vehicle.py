"""The interface every Basilisk vehicle meets, and the simulation each vehicle owns."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

__all__ = ["Vehicle"]

_TASK = "vehicle_task"


class Vehicle(ABC):
    """A spacecraft on its own Basilisk simulation, process and task.

    The base holds what every vehicle shares and the plugin and runner rely on: the simulation,
    process and task, the run-once rule, readers that keep their vehicle alive, and the declared
    `mu`, standard dispersion, initial state and recorded fields.

    A subclass takes its parameters as keywords defaulting to class attributes of the same name,
    calls `super().__init__(step_s=...)` and adds its models with `add_model`. The messages in
    `recorded_fields()` and the Basilisk objects a subclass exposes are valid while the vehicle is
    referenced: a message holding its vehicle would form a loop through Basilisk's C++ objects
    that the garbage collector leaves in place.

    Attributes:
        mu: The central body's gravitational parameter in the vehicle's gravity model, in m^3/s^2.
        standard_dispersion: Keyword to its per-axis 1-sigma, drawn in this order.
    """

    mu: ClassVar[float]
    standard_dispersion: ClassVar[Mapping[str, float]]

    def __init__(self, *, step_s: float) -> None:
        """Create the vehicle's simulation, with one process holding one task.

        Args:
            step_s: The task's period, in seconds: the dynamics and sensor step.

        Raises:
            ValueError: `step_s` is not positive.
        """
        if not step_s > 0:
            raise ValueError(f"step must be positive, got {step_s} s")
        self.step_s = step_s
        self._simulation = SimulationBaseClass.SimBaseClass()
        process = self._simulation.CreateNewProcess("vehicle_process")
        process.addTask(self._simulation.CreateNewTask(_TASK, macros.sec2nano(step_s)))
        self._executed = False

    def readers(self) -> dict[str, Any]:
        """The vehicle's sensor outputs by role, each a Basilisk message reader subscribed to its
        sensor, for a flight-software module to read."""
        handed_out = self._readers()
        for reader in handed_out.values():
            reader._vehicle = self
        return handed_out

    @abstractmethod
    def _readers(self) -> dict[str, Any]:
        """The sensor readers `readers()` returns, built afresh on each call."""

    @abstractmethod
    def recorded_fields(self) -> dict[str, tuple[Any, str, str]]:
        """Run-file dataset path to the `(message, field, units)` it records from this vehicle."""

    @abstractmethod
    def initial_state(self) -> dict[str, np.ndarray]:
        """The vehicle's initial condition as the simulation holds it, by key. A subclass
        documents its keys."""

    def add_model(self, model: Any) -> None:
        """Add a Basilisk module or recorder to the vehicle's task, after the models already there.

        Raises:
            RuntimeError: The vehicle has already run.
        """
        self._check_unexecuted()
        self._simulation.AddModelToTask(_TASK, model)

    def execute(self, duration_s: float) -> None:
        """Initialize the simulation and run it from t = 0 for `duration_s` seconds.

        Raises:
            ValueError: `duration_s` is not positive.
            RuntimeError: The vehicle has already run.
        """
        self._check_unexecuted()
        if not duration_s > 0:
            raise ValueError(f"duration must be positive, got {duration_s} s")
        self._executed = True
        self._simulation.InitializeSimulation()
        self._simulation.ConfigureStopTime(macros.sec2nano(duration_s))
        self._simulation.ExecuteSimulation()

    def _check_unexecuted(self) -> None:
        if self._executed:
            raise RuntimeError("this vehicle has already run; build a new vehicle for another run")
