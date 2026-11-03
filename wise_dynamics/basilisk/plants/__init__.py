"""The registry of Basilisk plants, by the name `--sim-spacecraft` uses.

A plant joins as a package here, one class per module (its configuration and its `Plant`
subclass), and one entry in `PLANTS`.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from wise_dynamics.basilisk.plant import Plant
from wise_dynamics.basilisk.plants.imu_spacecraft import ImuSpacecraft

__all__ = ["PLANTS"]

PLANTS: Mapping[str, type[Plant[Any]]] = MappingProxyType({"imu_spacecraft": ImuSpacecraft})
