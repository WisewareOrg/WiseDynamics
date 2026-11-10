"""Sessions: `session_directory`, the per-case result and crash record, and `read_test_results`."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest

NO_SESSION = "no Monte Carlo session is active: run pytest --sim-session <name>"


def _sessions() -> ModuleType:
    """Import `wise_dynamics.sessions` when a test needs it, so its absence fails that test only.

    Returns:
        The module.
    """
    return importlib.import_module("wise_dynamics.sessions")


def _point(pytester: pytest.Pytester, name: str, root: str = "monte_carlo") -> Path:
    """Write the session pointer and return the session's folder.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        name: What the pointer holds, as written.
        root: The `sim_out` root, relative to the rootdir.

    Returns:
        `<rootdir>/<root>/<name>`.
    """
    (pytester.path / root).mkdir(parents=True, exist_ok=True)
    (pytester.path / root / "SESSION").write_text(name)
    return pytester.path / root / name.strip()
