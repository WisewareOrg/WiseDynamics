"""`VEHICLES`: each vehicles module registers its one vehicle under the module's name."""

import re
import types
from collections.abc import Callable
from pathlib import Path

import pytest

import wise_dynamics.basilisk.vehicles as vehicles_package
from wise_dynamics.basilisk import ImuSpacecraft
from wise_dynamics.basilisk.vehicles import VEHICLES
from wise_dynamics.basilisk.vehicles.heavy_imu_spacecraft import HeavyImuSpacecraft


@pytest.mark.parametrize(
    "source",
    [
        "X = 1\n",
        "from wise_dynamics.basilisk.vehicle import Vehicle\n"
        "class First(Vehicle):\n    pass\n"
        "class Second(Vehicle):\n    pass\n",
    ],
    ids=["none", "two"],
)
def test_a_module_not_defining_exactly_one_vehicle_subclass_raises(
    make_package: Callable[[str], types.ModuleType], source: str
):
    """A vehicles module defining no `Vehicle` subclass, or two, raises `TypeError` naming it.

    Args:
        make_package: Builds a one-module package from source.
        source: The module's source.
    """
    package = make_package(source)
    with pytest.raises(TypeError, match=re.escape(f"{package.__name__}.mod")):
        vehicles_package._vehicles_in(package)


def test_a_vehicle_class_a_module_imports_is_not_counted(
    make_package: Callable[[str], types.ModuleType],
):
    """A module importing `ImuSpacecraft` and defining `Mine(ImuSpacecraft)` registers only `Mine`.

    Args:
        make_package: Builds a one-module package from source.
    """
    source = (
        "from wise_dynamics.basilisk.vehicles.imu_spacecraft import ImuSpacecraft\n"
        "class Mine(ImuSpacecraft):\n    pass\n"
    )
    package = make_package(source)
    assert vehicles_package._vehicles_in(package) == {"mod": package.mod.Mine}


def test_every_module_in_the_package_is_registered_under_its_name():
    """The vehicles directory's modules are the keys of `VEHICLES`, each its module's class."""
    directory = Path(vehicles_package.__file__).parent
    stems = {path.stem for path in directory.glob("*.py")} - {"__init__"}
    assert stems == set(VEHICLES)
    assert VEHICLES["imu_spacecraft"] is ImuSpacecraft
    assert VEHICLES["heavy_imu_spacecraft"] is HeavyImuSpacecraft
