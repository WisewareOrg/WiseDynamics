"""Basilisk vehicles and scenarios.

`README.md` beside this file is the reference for the package.
"""

from wise_dynamics.basilisk.fsw_host import FswBuild, FswHost
from wise_dynamics.basilisk.scenario import Scenario
from wise_dynamics.basilisk.scenarios import SCENARIOS
from wise_dynamics.basilisk.vehicle import Vehicle
from wise_dynamics.basilisk.vehicles import VEHICLES
from wise_dynamics.basilisk.vehicles.imu_spacecraft import ImuSpacecraft

__all__ = [
    "SCENARIOS",
    "VEHICLES",
    "FswBuild",
    "FswHost",
    "ImuSpacecraft",
    "Scenario",
    "Vehicle",
]
