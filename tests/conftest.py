"""Enables pytest's own `pytester` fixture and shares test-only fixtures."""

import importlib
import os
import sys
import textwrap
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


def pytest_configure(config: pytest.Config) -> None:
    """Register the `capped_runs` mark.

    Args:
        config: The session's pytest configuration.
    """
    config.addinivalue_line(
        "markers",
        "capped_runs: the test's pytester sessions run as given, without the "
        "`--full-monte-carlo` flag the other tests' sessions are started with",
    )


@pytest.fixture(autouse=True)
def full_flights_by_default(request: pytest.FixtureRequest) -> None:
    """Start every pytester subprocess session with `--full-monte-carlo`, unless the test opts out.

    A simulator session flies at most two cases per test unless it is given the flag, so a test
    that checks more cases, or where the runs land in the active session, needs it. A test marked
    `capped_runs` is checking the capped behaviour and passes its own arguments unchanged.

    Args:
        request: The fixture request, used to reach `pytester` and the test's marks.
    """
    if "pytester" not in request.fixturenames or request.node.get_closest_marker("capped_runs"):
        return
    pytester = request.getfixturevalue("pytester")
    run = pytester.runpytest_subprocess

    def run_full(*args: str | os.PathLike[str], timeout: float | None = None) -> pytest.RunResult:
        """Run a subprocess session with `--full-monte-carlo` first among the arguments.

        Args:
            *args: The session's pytest arguments.
            timeout: Seconds after which the session is stopped.

        Returns:
            The session's result.
        """
        return run("--full-monte-carlo", *args, timeout=timeout)

    pytester.runpytest_subprocess = run_full


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
