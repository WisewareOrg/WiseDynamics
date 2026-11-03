"""The registry of Basilisk plants, by the name `--sim-spacecraft` uses.

A plant joins as two modules here, `<plant>.py` holding its `Plant` subclass and
`<plant>_config.py` holding its configuration, and one entry in `PLANTS`.
"""

from collections.abc import Mapping
from types import MappingProxyType

from wise_dynamics.basilisk.plant import Plant
from wise_dynamics.basilisk.plants.imu_spacecraft import ImuSpacecraft

__all__ = ["PLANTS"]

PLANTS: Mapping[str, type[Plant]] = MappingProxyType({"imu_spacecraft": ImuSpacecraft})
