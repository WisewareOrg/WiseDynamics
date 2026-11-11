"""Enables pytest's own `pytester` fixture and shares test-only fixtures."""

import importlib
import sys
import textwrap
import types
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

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
