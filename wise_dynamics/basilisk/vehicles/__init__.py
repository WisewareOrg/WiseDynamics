"""The registry of Basilisk vehicles, by the name the `sim_vehicle` mark uses.

Each module here defines one `Vehicle` subclass and is registered in `VEHICLES` under the module's
name. Importing the package imports every module here.
"""

import importlib
import pkgutil
import sys

from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["VEHICLES"]


def _vehicles_in(package: object) -> dict[str, type[Vehicle]]:
    """Map each module of a package to the one `Vehicle` subclass it defines.

    Args:
        package: The package whose modules are imported and searched.

    Returns:
        The vehicle class of each module, keyed by the module's name.

    Raises:
        TypeError: A module defines no `Vehicle` subclass, or more than one.
    """
    vehicles: dict[str, type[Vehicle]] = {}
    for info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        defined = [
            value
            for value in vars(module).values()
            if isinstance(value, type)
            and issubclass(value, Vehicle)
            and value.__module__ == module.__name__
        ]
        if len(defined) != 1:
            raise TypeError(f"vehicle module {module.__name__} defines {len(defined)} vehicles")
        vehicles[info.name] = defined[0]
    return vehicles


VEHICLES: dict[str, type[Vehicle]] = _vehicles_in(sys.modules[__name__])
