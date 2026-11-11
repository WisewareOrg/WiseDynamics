"""The registry of Basilisk scenarios, by the name the `sim_scenario` mark uses.

Each module here defines one `Scenario` subclass and is registered in `SCENARIOS` under the module's
name. Importing the package imports every module here.
"""

import sys
import types

from wise_dynamics.basilisk._registry import subclasses_by_module
from wise_dynamics.basilisk.scenario import Scenario

__all__ = ["SCENARIOS"]


def _scenarios_in(package: types.ModuleType) -> dict[str, type[Scenario]]:
    """Map each module of a package to the one `Scenario` subclass it defines.

    Args:
        package: The package whose modules are imported and searched.

    Returns:
        The scenario class of each module, keyed by the module's name.
    """
    return subclasses_by_module(package, Scenario)


SCENARIOS: dict[str, type[Scenario]] = _scenarios_in(sys.modules[__name__])
