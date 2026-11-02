"""The registry of Basilisk plants, by the name a campaign and `--sim-spacecraft` use.

A plant joins by subclassing `wise_dynamics.basilisk.Plant` in its own module here and adding one
entry to `PLANTS`.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from wise_dynamics.basilisk.plant import Plant
from wise_dynamics.basilisk.plants.imu_spacecraft import ImuSpacecraft

__all__ = ["PLANTS"]

PLANTS: Mapping[str, type[Plant[Any]]] = MappingProxyType({"imu_spacecraft": ImuSpacecraft})
