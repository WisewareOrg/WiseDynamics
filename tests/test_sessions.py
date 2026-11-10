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


def test_session_directory_is_the_root_joined_to_the_pointers_name(pytester: pytest.Pytester):
    """The pointer's stripped content names the folder under the `sim_out` root.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    _point(pytester, "alpha-1.2\n")

    assert _sessions().session_directory(pytester.parseconfig()) == (
        pytester.path / "monte_carlo" / "alpha-1.2"
    )


def test_session_directory_follows_the_sim_out_ini_setting(pytester: pytest.Pytester):
    """A relative `sim_out` is resolved against the rootdir.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
    """
    pytester.makeini("[pytest]\nsim_out = ini_out\n")
    _point(pytester, "beta", root="ini_out")

    assert _sessions().session_directory(pytester.parseconfig()) == (
        pytester.path / "ini_out" / "beta"
    )


@pytest.mark.parametrize("content", [None, "", "  \n"], ids=["missing", "empty", "blank"])
def test_session_directory_with_no_active_session_raises_a_usage_error(
    pytester: pytest.Pytester, content: str | None
):
    """A missing, empty or blank pointer raises `UsageError` with the no-session message.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        content: What `SESSION` holds, or `None` for no file.
    """
    if content is not None:
        _point(pytester, content)
    config = pytester.parseconfig()

    with pytest.raises(pytest.UsageError) as raised:
        _sessions().session_directory(config)

    assert str(raised.value) == NO_SESSION


@pytest.mark.parametrize("name", ["..", "-x", "SESSION", ".hidden", "a/b", "a b"])
def test_session_directory_with_an_invalid_name_raises_a_usage_error_naming_the_file(
    pytester: pytest.Pytester, name: str
):
    """An invalid name in the pointer raises `UsageError` naming `SESSION`, not the no-session one.

    Args:
        pytester: pytest's own fixture for running a test session in a subprocess.
        name: The invalid name the pointer holds.
    """
    _point(pytester, name + "\n")
    config = pytester.parseconfig()

    with pytest.raises(pytest.UsageError) as raised:
        _sessions().session_directory(config)

    assert "SESSION" in str(raised.value)
    assert str(raised.value) != NO_SESSION
