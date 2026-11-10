"""Monte Carlo sessions: the named working sets that a campaign's run files are kept in.

The `sim_out` ini setting names the root. `<root>/SESSION` holds the active session's name, and
each session is the folder `<root>/<name>`. `wise_dynamics/basilisk/README.md` documents the layout.
"""

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

import pytest

__all__ = ["TestResults", "read_test_results", "session_directory"]

POINTER = "SESSION"
PLANNED = "planned.json"
RUN_GLOB = "run_*.h5"
NO_SESSION = "no Monte Carlo session is active: run pytest --sim-session <name>"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class TestResults(NamedTuple):
    """What a Monte Carlo test left in its session directory.

    Attributes:
        planned: The number of cases the test planned.
        runs: Case number to the case's run file, for the cases below `planned` that have one.
        crashes: Case number to the case's crash record, for the cases below `planned` that have
            one.
    """

    __test__ = False

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


def session_directory(config: pytest.Config) -> Path:
    """Find the active session's directory.

    Args:
        config: The pytest configuration whose rootdir and `sim_out` ini setting locate the root.

    Returns:
        `<root>/<name>`, `name` being the stripped content of `<root>/SESSION`. The directory may
        not exist yet.

    Raises:
        pytest.UsageError: The pointer is missing or empty, or the name it holds is invalid.
    """
    root = sim_root(config)
    pointer = root / POINTER
    name = pointer.read_text().strip() if pointer.is_file() else ""
    if not name:
        raise pytest.UsageError(NO_SESSION)
    return root / check_name(name, str(pointer))


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


def read_test_results(directory: Path) -> TestResults:
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
    return TestResults(
        planned,
        {case: path for case, path in runs.items() if path.is_file()},
        {case: json.loads(path.read_text()) for case, path in crashes.items() if path.is_file()},
    )
