"""`SCENARIOS`: each scenarios module registers its one scenario under the module's name."""

import sys
import textwrap
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import wise_dynamics.basilisk as basilisk
import wise_dynamics.basilisk.scenarios as scenarios_package
from wise_dynamics.basilisk.scenario import Scenario
from wise_dynamics.basilisk.scenarios import SCENARIOS

SCENARIOS_PACKAGE = "wise_dynamics.basilisk.scenarios"
SUBCLASS = "class {name}(Scenario):\n    pass\n"


@pytest.fixture
def make_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[Callable[[str], types.ModuleType]]:
    """Build a one-module package under `tmp_path`, importable and named uniquely for the test.

    Its `sys.modules` entries are removed at teardown.

    Args:
        tmp_path: The test's temporary directory.
        monkeypatch: Puts `tmp_path` on `sys.path` for the test.
        request: Names the package after the requesting test.

    Yields:
        Callable[[str], types.ModuleType]: Takes the module's source, returns the imported package.
    """
    name = f"pkg_{request.node.name}"
    monkeypatch.syspath_prepend(str(tmp_path))

    def make(source: str) -> types.ModuleType:
        """Write the package with one module, `mod`, holding `source`, and import it.

        Args:
            source: The module's source.

        Returns:
            The imported package.
        """
        package_dir = tmp_path / name
        package_dir.mkdir()
        (package_dir / "__init__.py").write_text("")
        (package_dir / "mod.py").write_text(textwrap.dedent(source))
        __import__(name)
        return sys.modules[name]

    yield make
    for key in [k for k in sys.modules if k == name or k.startswith(f"{name}.")]:
        del sys.modules[key]


def test_a_module_with_no_scenario_subclass_raises(make_package: Callable[[str], types.ModuleType]):
    """A scenarios module defining no `Scenario` subclass raises `TypeError` naming the module.

    Args:
        make_package: Builds a one-module package from source.
    """
    package = make_package("X = 1\n")
    with pytest.raises(TypeError, match=rf"{package.__name__}\.mod"):
        scenarios_package._scenarios_in(package)


def test_a_module_with_two_scenario_subclasses_raises(
    make_package: Callable[[str], types.ModuleType],
):
    """A scenarios module defining two `Scenario` subclasses raises `TypeError` naming the module.

    Args:
        make_package: Builds a one-module package from source.
    """
    source = (
        "from wise_dynamics.basilisk.scenario import Scenario\n"
        + SUBCLASS.format(name="First")
        + SUBCLASS.format(name="Second")
    )
    package = make_package(source)
    with pytest.raises(TypeError, match=rf"{package.__name__}\.mod"):
        scenarios_package._scenarios_in(package)


def test_every_module_in_the_package_is_registered_under_its_name():
    """The scenarios directory's modules are the keys of `SCENARIOS`, each defining its entry."""
    directory = Path(scenarios_package.__file__).parent
    stems = {path.stem for path in directory.glob("*.py")} - {"__init__"}
    assert stems == set(SCENARIOS)
    for name, scenario in SCENARIOS.items():
        assert scenario.__module__ == f"{SCENARIOS_PACKAGE}.{name}"


def test_the_package_exports_no_scenario_class():
    """`wise_dynamics.basilisk.__all__` names no `Scenario` subclass but `Scenario` itself."""
    scenarios = [
        name
        for name in basilisk.__all__
        if isinstance(getattr(basilisk, name), type)
        and issubclass(getattr(basilisk, name), Scenario)
        and getattr(basilisk, name) is not Scenario
    ]
    assert scenarios == []
