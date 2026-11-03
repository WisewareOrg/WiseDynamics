"""The shape every plant configuration has."""

from typing import Protocol

from wise_dynamics.dataclass_config import DataclassConfig

__all__ = ["PlantConfig"]


class PlantConfig(DataclassConfig, Protocol):
    """A plant's configuration: a dataclass carrying the plant's step."""

    @property
    def step_s(self) -> float:
        """The dynamics and sensor step, in seconds."""
        ...
