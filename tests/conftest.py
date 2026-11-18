"""Enables pytest's own `pytester` fixture and shares test-only fixtures."""

import importlib
import sys
import textwrap
import types
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from Basilisk.architecture import messaging

pytest_plugins = ["pytester"]


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
        return importlib.import_module(name)

    yield make
    for key in [k for k in sys.modules if k == name or k.startswith(f"{name}.")]:
        del sys.modules[key]


class _Commanding:
    """A build returning `commands` at each of its first `ticks` ticks, `{}` after.

    It counts the ticks it has run in `ticked`.

    Attributes:
        period_s: The build's period, in seconds.
    """

    period_s = 0.5

    def __init__(self, commands: Mapping[str, Any], ticks: int | None = None) -> None:
        """Set the commands and for how many ticks they are returned.

        Args:
            commands: What each of the first `ticks` ticks returns.
            ticks: The ticks returning `commands`, every tick when `None`; later ticks return `{}`.
        """
        self.commands = commands
        self.ticks = ticks
        self.ticked = 0

    def boot(self) -> None:
        """Start, with nothing to do."""

    def tick(self, t_ns: int, messages_by_role: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return `commands` for the first `ticks` ticks, `{}` after.

        Args:
            t_ns: The tick's simulation time, in nanoseconds.
            messages_by_role: The vehicle's readings by role.

        Returns:
            The tick's commands.
        """
        self.ticked += 1
        return self.commands if self.ticks is None or self.ticked <= self.ticks else {}


@pytest.fixture
def commanding() -> type[_Commanding]:
    """The stub build that commands what it is given, for a host test or a vehicle's.

    Returns:
        The class: `commanding(commands, ticks=None)` returns `commands` at each of its first
        `ticks` ticks, every tick when `ticks` is `None`, and `{}` after; `ticked` counts its
        ticks.
    """
    return _Commanding


def _torque(value: float) -> Any:
    """An `ArrayMotorTorqueMsgPayload` whose first torque is `value`.

    Args:
        value: The first wheel's torque, in N*m.

    Returns:
        The payload.
    """
    payload = messaging.ArrayMotorTorqueMsgPayload()
    payload.motorTorque = [value] + [0.0] * (messaging.MAX_EFF_CNT - 1)
    return payload


@pytest.fixture
def torque() -> Callable[[float], Any]:
    """Builds a wheel torque command.

    Returns:
        A function taking the first wheel's torque, in N*m, and giving the
        `ArrayMotorTorqueMsgPayload` that commands it.
    """
    return _torque
