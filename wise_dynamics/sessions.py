"""Monte Carlo sessions: the named working sets that a campaign's run files are kept in.

The `sim_out` ini setting names the root. `<root>/SESSION` holds the active session's name, and
each session is the folder `<root>/<name>`. `wise_dynamics/basilisk/README.md` documents the layout.
"""

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import pytest

__all__ = [
    "CaseResults",
    "NoSessionError",
    "RecordedTest",
    "crash_path",
    "read_test_results",
    "recorded_tests",
    "run_path",
    "session_directory",
    "write_planned",
]

POINTER = "SESSION"
PLANNED = "planned.json"
NO_SESSION = "no Monte Carlo session is active: run pytest --sim-session <name>"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class NoSessionError(pytest.UsageError):
    """No Monte Carlo session is active."""


class RecordedTest(NamedTuple):
    """A test's results folder in a session.

    Attributes:
        directory: The test's directory, `<session>/<file>/<class>/<function>`.
        nodeid: The test's node path relative to the rootdir, `<file>.py::<Class>::<function>`,
            without parameters.
    """

    directory: Path
    nodeid: str


class CaseResults(NamedTuple):
    """What a Monte Carlo test left in its session directory.

    Attributes:
        planned: The number of cases the test planned.
        runs: Case number to the case's run file, for the cases below `planned` that have one.
        crashes: Case number to the case's crash record, for the cases below `planned` that have
            one.
    """

    planned: int
    runs: dict[int, Path]
    crashes: dict[int, dict[str, Any]]


def sim_root(config: pytest.Config) -> Path:
    """Resolve the directory that holds the pointer and the sessions.

    Args:
        config: The pytest configuration, parsed but not necessarily configured.

    Returns:
        The `sim_out` ini setting, joined to the rootdir when relative.

    Raises:
        pytest.UsageError: The `sim_out` ini setting is empty.
    """
    out = config.getini("sim_out")
    if not out:
        raise pytest.UsageError("the sim_out ini setting must not be empty")
    return config.rootpath / out


def check_name(name: str, source: str) -> str:
    """Validate a session name.

    Args:
        name: The candidate name.
        source: Where the name came from, for the error message.

    Returns:
        `name`.

    Raises:
        pytest.UsageError: `name` is not a letter or digit followed by letters, digits, `.`, `_`
            and `-`, or is `SESSION`.
    """
    if not _NAME.fullmatch(name) or name == POINTER:
        raise pytest.UsageError(
            f"invalid Monte Carlo session name {name!r} in {source}: start with a letter or "
            f"digit, then use letters, digits, '.', '_' and '-'; {POINTER} is reserved"
        )
    return name


def read_pointer(root: Path) -> str:
    """Read the active session's name from the pointer.

    Args:
        root: The `sim_out` root.

    Returns:
        The pointer's stripped content, empty when the pointer is missing or empty.
    """
    pointer = root / POINTER
    return pointer.read_text().strip() if pointer.is_file() else ""


def write_pointer(root: Path, name: str) -> None:
    """Make `name` the active session, creating the root if needed.

    Args:
        root: The `sim_out` root.
        name: A valid session name.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / POINTER).write_text(f"{name}\n")


def session_directory(config: pytest.Config) -> Path:
    """Find the active session's directory.

    Args:
        config: The pytest configuration whose rootdir and `sim_out` ini setting locate the root.

    Returns:
        `<root>/<name>`, `name` being the stripped content of `<root>/SESSION`. The directory may
        not exist yet.

    Raises:
        NoSessionError: The pointer is missing or empty. A subclass of `pytest.UsageError`, which
            is also raised when the name the pointer holds is invalid.
    """
    root = sim_root(config)
    name = read_pointer(root)
    if not name:
        raise NoSessionError(NO_SESSION)
    return root / check_name(name, str(root / POINTER))


def run_path(directory: Path, case: int) -> Path:
    """Name a case's run file.

    Args:
        directory: A test's directory in a session.
        case: The case number.

    Returns:
        The case's run file path.
    """
    return directory / f"run_{case:04d}.h5"


def crash_path(directory: Path, case: int) -> Path:
    """Name a case's crash record.

    Args:
        directory: A test's directory in a session.
        case: The case number.

    Returns:
        The case's crash record path.
    """
    return directory / f"run_{case:04d}.crash.json"


def write_planned(directory: Path, runs: int, nodeid: str) -> None:
    """Record a test's planned case count and node path, creating its directory if needed.

    Args:
        directory: The test's directory in a session.
        runs: The test's case count after the `--ci` cap.
        nodeid: The test's node path relative to the rootdir, without parameters.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PLANNED).write_text(json.dumps({"runs": runs, "nodeid": nodeid}))


def recorded_tests(session: Path) -> list[RecordedTest]:
    """List the test folders of a session that hold results.

    Args:
        session: A session's directory.

    Returns:
        Each folder under `session` whose record names its test, sorted by directory.
    """
    found = []
    for record in session.rglob(PLANNED):
        nodeid = json.loads(record.read_text()).get("nodeid")
        if nodeid is not None:
            found.append(RecordedTest(record.parent, nodeid))
    return sorted(found)


def read_test_results(directory: Path) -> CaseResults:
    """Read what a Monte Carlo test left in its session directory.

    Args:
        directory: The test's directory in a session, `<session>/<file>/<class>/<function>`.

    Returns:
        The planned case count, and the run files and crash records of cases `0..planned-1`.
        Files of higher case numbers are ignored.
    """
    planned = json.loads((directory / PLANNED).read_text())["runs"]
    runs = {case: run_path(directory, case) for case in range(planned)}
    crashes = {case: crash_path(directory, case) for case in range(planned)}
    return CaseResults(
        planned,
        {case: path for case, path in runs.items() if path.is_file()},
        {case: json.loads(path.read_text()) for case, path in crashes.items() if path.is_file()},
    )
