"""The registry of Basilisk vehicles, by the name `--sim-vehicle` uses.

A vehicle joins as a module here, `<vehicle>.py` holding its `Vehicle` subclass, and one entry in
`VEHICLES`.
"""

from collections.abc import Mapping
from types import MappingProxyType

from wise_dynamics.basilisk.vehicle import Vehicle
from wise_dynamics.basilisk.vehicles.imu_spacecraft import ImuSpacecraft

__all__ = ["VEHICLES"]

VEHICLES: Mapping[str, type[Vehicle]] = MappingProxyType({"imu_spacecraft": ImuSpacecraft})
