"""`SCENARIOS`: each scenarios module registers its one scenario under the module's name."""

import re
import types
from collections.abc import Callable
from pathlib import Path

import pytest

import wise_dynamics.basilisk as basilisk
import wise_dynamics.basilisk.scenarios as scenarios_package
from wise_dynamics.basilisk.scenario import Scenario
from wise_dynamics.basilisk.scenarios import SCENARIOS


@pytest.mark.parametrize(
    "source",
    [
        "X = 1\n",
        "from wise_dynamics.basilisk.scenario import Scenario\n"
        "class First(Scenario):\n    pass\n"
        "class Second(Scenario):\n    pass\n",
    ],
    ids=["none", "two"],
)
def test_a_module_not_defining_exactly_one_scenario_subclass_raises(
    make_package: Callable[[str], types.ModuleType], source: str
):
    """A scenarios module defining no `Scenario` subclass, or two, raises `TypeError` naming it.

    Args:
        make_package: Builds a one-module package from source.
        source: The module's source.
    """
    package = make_package(source)
    with pytest.raises(TypeError, match=re.escape(f"{package.__name__}.mod")):
        scenarios_package._scenarios_in(package)


def test_a_scenario_class_a_module_imports_is_not_counted(
    make_package: Callable[[str], types.ModuleType],
):
    """A module importing `Leo500` and defining `Mine(Leo500)` registers `Mine`, and only `Mine`.

    Args:
        make_package: Builds a one-module package from source.
    """
    source = (
        "from wise_dynamics.basilisk.scenarios.leo500 import Leo500\n"
        "class Mine(Leo500):\n    pass\n"
    )
    package = make_package(source)
    assert scenarios_package._scenarios_in(package) == {"mod": package.mod.Mine}


def test_every_module_in_the_package_is_registered_under_its_name():
    """The scenarios directory's modules are the keys of `SCENARIOS`."""
    directory = Path(scenarios_package.__file__).parent
    stems = {path.stem for path in directory.glob("*.py")} - {"__init__"}
    assert stems == set(SCENARIOS)


def test_the_package_exports_no_scenario_class():
    """`wise_dynamics.basilisk.__all__` names no `Scenario` subclass but `Scenario` itself."""
    exported = [getattr(basilisk, name) for name in basilisk.__all__]
    scenarios = [
        value
        for value in exported
        if isinstance(value, type) and issubclass(value, Scenario) and value is not Scenario
    ]
    assert scenarios == []
