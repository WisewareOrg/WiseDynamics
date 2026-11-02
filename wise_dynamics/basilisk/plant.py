"""The contract every Basilisk plant meets, and the simulation each plant owns."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

from wise_dynamics.campaign import distribution_version
from wise_dynamics.dispersion import Dispersion

__all__ = ["Plant", "RecordedField"]

# Basilisk's SWIG-generated message, reader and module classes are typed as `Any`.
type Message = Any
type MessageReader = Any
type SysModel = Any


@dataclass(frozen=True)
class RecordedField:
    """One field of one recorded message, and the units it is expressed in.

    Attributes:
        role: The message's role, a key of `Plant.truth_messages()` or of the flight software's
            outputs handed to `run_case`.
        field: The payload field the recorder exposes, for example `"r_BN_N"`.
        units: The units string a run file stores with it, for example `"m"`.
    """

    role: str
    field: str
    units: str


class Plant[ConfigT](ABC):
    """A spacecraft plant on its own Basilisk simulation, process and task.

    A subclass builds its models in its constructor, which has the signature
    `__init__(self, config: ConfigT | None = None, *, noise_seed: int = 0)`, with `None`
    selecting `nominal()`. A plant runs once: `execute` initializes and runs the simulation,
    and a new plant is built for the next run.

    Class attributes a subclass sets:
        standard_dispersion: The vehicle's standard per-case dispersion of its configuration.
        recorded_fields: Run-file dataset name to the truth message field it records, with
            units, covering every field of `truth_messages()` a run file carries.
    """

    standard_dispersion: ClassVar[Dispersion]
    recorded_fields: ClassVar[Mapping[str, RecordedField]]

    def __init__(self, config: ConfigT, *, step_s: float) -> None:
        """Create the plant's simulation, with one process holding one task.

        Args:
            config: The configuration the subclass builds from, kept as `config`.
            step_s: The task's period, in seconds: the dynamics, sensor and flight-software
                step.

        Raises:
            ValueError: `step_s` is not positive.
        """
        if not step_s > 0:
            raise ValueError(f"step must be positive, got {step_s} s")
        self._config = config
        self._simulation = SimulationBaseClass.SimBaseClass()
        self._task_name = "plant_task"
        process = self._simulation.CreateNewProcess("plant_process")
        process.addTask(self._simulation.CreateNewTask(self._task_name, macros.sec2nano(step_s)))
        self._executed = False

    @classmethod
    @abstractmethod
    def nominal(cls) -> ConfigT:
        """The vehicle's nominal configuration."""

    @property
    def config(self) -> ConfigT:
        """The configuration this plant was built from."""
        return self._config

    @property
    def simulation(self) -> SimulationBaseClass.SimBaseClass:
        """The Basilisk simulation this plant owns."""
        return self._simulation

    @property
    def task_name(self) -> str:
        """The name of the one task every model of this plant runs in."""
        return self._task_name

    @property
    @abstractmethod
    def mu(self) -> float:
        """The central body's gravitational parameter, in m^3/s^2."""

    @abstractmethod
    def readers(self) -> dict[str, MessageReader]:
        """The plant's sensor outputs by role, each a Basilisk message reader already subscribed
        to its sensor, for a flight-software module to read."""

    @abstractmethod
    def truth_messages(self) -> dict[str, Message]:
        """The plant's recordable output messages by role."""

    @abstractmethod
    def initial_state(self) -> dict[str, np.ndarray]:
        """The vehicle's initial condition as the simulation holds it, by settled key."""

    @classmethod
    def provenance(cls) -> dict[str, str]:
        """Run-file root attributes naming what produced a run of this plant.

        Returns:
            `producer`, the host package's name, and `versions_bsk`, the installed Basilisk
            version.
        """
        return {"producer": "wise_dynamics.basilisk", "versions_bsk": distribution_version("bsk")}

    def add_model(self, model: SysModel) -> None:
        """Add a Basilisk module or recorder to the plant's task, after the models already there.

        Raises:
            RuntimeError: The plant has already run.
        """
        self._check_unexecuted()
        self._simulation.AddModelToTask(self._task_name, model)

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
