"""The registry of Basilisk scenarios, by the name the `sim_scenario` mark uses.

Each module here defines one `Scenario` subclass and is registered in `SCENARIOS` under the module's
name. Importing the package imports every module here.
"""

import importlib
import pkgutil
import sys
import types

from wise_dynamics.basilisk.scenario import Scenario

__all__ = ["SCENARIOS"]


def _scenarios_in(package: types.ModuleType) -> dict[str, type[Scenario]]:
    """Map each module of a package to the one `Scenario` subclass it defines.

    Args:
        package: The package whose modules are imported and searched.

    Returns:
        The scenario class of each module, keyed by the module's name.

    Raises:
        TypeError: A module defines no `Scenario` subclass, or more than one. Classes a module
            imports do not count.
    """
    scenarios: dict[str, type[Scenario]] = {}
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        defined = [
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, Scenario)
            and value is not Scenario
            and value.__module__ == module.__name__
        ]
        if len(defined) != 1:
            raise TypeError(
                f"scenario module {module.__name__} defines {len(defined)} Scenario subclasses; "
                "it must define exactly one"
            )
        scenarios[info.name] = defined[0]
    return scenarios


SCENARIOS: dict[str, type[Scenario]] = _scenarios_in(sys.modules[__name__])
