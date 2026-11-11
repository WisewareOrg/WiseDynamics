"""The registry of Basilisk vehicles, by the name the `sim_vehicle` mark uses.

Each module here defines one `Vehicle` subclass and is registered in `VEHICLES` under the module's
name. Importing the package imports every module here.
"""

import sys
import types

from wise_dynamics.basilisk._registry import subclasses_by_module
from wise_dynamics.basilisk.vehicle import Vehicle

__all__ = ["VEHICLES"]


def _vehicles_in(package: types.ModuleType) -> dict[str, type[Vehicle]]:
    """Map each module of a package to the one `Vehicle` subclass it defines.

    Args:
        package: The package whose modules are imported and searched.

    Returns:
        The vehicle class of each module, keyed by the module's name.
    """
    return subclasses_by_module(package, Vehicle)


VEHICLES: dict[str, type[Vehicle]] = _vehicles_in(sys.modules[__name__])
