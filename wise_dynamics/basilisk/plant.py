"""The interface every Basilisk plant meets, and the simulation each plant owns."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

from wise_dynamics.basilisk.recorded_field import RecordedField
from wise_dynamics.dispersion import Dispersion

__all__ = ["Plant"]

_TASK = "plant_task"


class Plant(ABC):
    """A spacecraft plant on its own Basilisk simulation, process and task.

    A plant runs once: `execute` initializes and runs the simulation, and a new plant is built
    for the next run. Each reader `readers()` returns holds its plant, so it reads the plant's
    live output for as long as it is held. The messages `truth_messages()` returns and the
    Basilisk objects a subclass exposes as attributes, along with anything reached through them,
    are valid while the plant is referenced: a recorder holds its message, so a message holding
    its plant would form a loop through Basilisk's C++ objects that the garbage collector leaves
    in place. A `Recording` holds its own data and outlives its plant.

    A subclass implements `nominal()`, `_build()`, `_readers()`, `truth_messages()` and
    `initial_state()`, and sets three class attributes:

    - `mu`: the central body's gravitational parameter its gravity model uses, in m^3/s^2.
    - `standard_dispersion`: the vehicle's standard per-case dispersion of its configuration.
    - `recorded_fields`: run-file dataset name to the truth message field it records, with
      units, covering every field of `truth_messages()` a run file carries.
    """

    mu: ClassVar[float]
    standard_dispersion: ClassVar[Dispersion]
    recorded_fields: ClassVar[Mapping[str, RecordedField]]

    def __init__(self, config: Any = None, *, noise_seed: int = 0) -> None:
        """Build the plant: its simulation, one process holding one task at `config.step_s`, and
        the subclass's models.

        Args:
            config: The configuration dataclass to build, carrying the step as `step_s` in
                seconds; `None` selects `nominal()`.
            noise_seed: Seed of the plant's sensor-noise generators.

        Raises:
            ValueError: `config.step_s` is not positive.
        """
        self._config = self.nominal() if config is None else config
        step_s = self._config.step_s
        if not step_s > 0:
            raise ValueError(f"step must be positive, got {step_s} s")
        self._simulation = SimulationBaseClass.SimBaseClass()
        process = self._simulation.CreateNewProcess("plant_process")
        process.addTask(self._simulation.CreateNewTask(_TASK, macros.sec2nano(step_s)))
        self._executed = False
        self._build(self._config, noise_seed)

    @classmethod
    @abstractmethod
    def nominal(cls) -> Any:
        """The vehicle's nominal configuration."""

    @abstractmethod
    def _build(self, config: Any, noise_seed: int) -> None:
        """Add the plant's models to its task, built from `config`, with sensor noise seeded by
        `noise_seed`."""

    @property
    def config(self) -> Any:
        """The configuration this plant was built from."""
        return self._config

    def readers(self) -> dict[str, Any]:
        """The plant's sensor outputs by role, each a Basilisk message reader already subscribed
        to its sensor, for a flight-software module to read. Each reader holds this plant."""
        handed_out = self._readers()
        for reader in handed_out.values():
            reader._plant = self
        return handed_out

    @abstractmethod
    def _readers(self) -> dict[str, Any]:
        """The sensor readers `readers()` returns, built afresh on each call."""

    @abstractmethod
    def truth_messages(self) -> dict[str, Any]:
        """The plant's recordable output messages by role."""

    @abstractmethod
    def initial_state(self) -> dict[str, np.ndarray]:
        """The vehicle's initial condition as the simulation holds it, by key. A subclass
        documents its keys."""

    def add_model(self, model: Any) -> None:
        """Add a Basilisk module or recorder to the plant's task, after the models already there.

        Raises:
            RuntimeError: The plant has already run.
        """
        self._check_unexecuted()
        self._simulation.AddModelToTask(_TASK, model)

    def execute(self, duration_s: float) -> None:
        """Initialize the simulation and run it from t = 0 for `duration_s`.

        Args:
            duration_s: Simulated time to run, in seconds.

        Raises:
            ValueError: `duration_s` is not positive.
            RuntimeError: The plant has already run.
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
            raise RuntimeError("this plant has already run; build a new plant for another run")
