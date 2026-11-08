"""The registry of Basilisk scenarios, by the name `--sim-scenario` uses.

A scenario joins as a module here, `<scenario>.py` holding its `Scenario` subclass, and one entry
in `SCENARIOS`.
"""

from wise_dynamics.basilisk.scenario import Scenario
from wise_dynamics.basilisk.scenarios.leo500 import Leo500

__all__ = ["SCENARIOS"]

SCENARIOS: dict[str, type[Scenario]] = {"leo500": Leo500}
