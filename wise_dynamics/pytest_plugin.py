"""The WiseDynamics pytest plugin: each test that uses a `sim_` fixture is one Monte Carlo campaign.

A test configures its campaign with marks on the function, a class or the module, the closest
winning. Installing `wise-dynamics` registers this module through its `pytest11` entry point.
`wise_dynamics/basilisk/README.md` documents the marks, `--full-monte-carlo` and fixtures.
"""

import json
import numbers
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import distribution, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.parse import urlparse
from urllib.request import url2pathname

import numpy as np
import pytest

from wise_dynamics import sessions

if TYPE_CHECKING:
    from wise_dynamics.basilisk.scenario import Scenario

__all__: list[str] = []

_WORKER_DIRS = "wise_dynamics_dirs"
_WORKER_SESSION = "wise_dynamics_session"
_SESSION = pytest.StashKey[Path]()
_WRITTEN_DIRS = pytest.StashKey[set[Path]]()
_PROVENANCE = pytest.StashKey[dict[str, str | bool]]()
_CALLER = pytest.StashKey[dict[str, str | bool]]()
_SEED = 0
_CASE_FILE = re.compile(r"run_(\d+)\.(?:h5|crash\.json)")
_INSTALL_HINT = (
    "Install a pinned release with `uv sync` against a git dependency, or a local checkout with\n"
    "`uv pip install -e <checkout>`."
)
_DEFAULT_OUT = "monte_carlo"
_CAPPED_RUNS = 2
_DEFAULT_SCENARIO = "leo500"
_DEFAULT_RUNS = 8
_DEFAULT_FIDELITY = "lo"
_DEFAULT_DURATION_S = 5700.0
_FIDELITIES = ("hi", "med", "lo")
_MARKS = (
    (
        "sim_scenario",
        "name",
        f"name of the scenario in wise_dynamics.basilisk.SCENARIOS; default {_DEFAULT_SCENARIO}",
    ),
    (
        "sim_vehicle",
        "name",
        "name of the vehicle in wise_dynamics.basilisk.vehicles.VEHICLES; default: the scenario's"
        " own",
    ),
    ("sim_runs", "n", f"number of cases, at least 1; default {_DEFAULT_RUNS}"),
    (
        "sim_fidelity",
        "level",
        "integration fidelity: hi is RK4 at 0.02 s, med RK4 at 0.05 s, lo RK2 at 0.1 s; "
        f"default {_DEFAULT_FIDELITY}",
    ),
    (
        "sim_duration_s",
        "seconds",
        f"simulated duration of each case, positive; default {_DEFAULT_DURATION_S:g}",
    ),
)


class _NoSession(NamedTuple):
    """The case parameter of a Monte Carlo test that has no usable session.

    Attributes:
        reason: Why there is none, shown as the test's setup error.
    """

    reason: str


class _Settings(NamedTuple):
    """One test's resolved configuration.

    Attributes:
        scenario: The scenario's name in `SCENARIOS`.
        runs: The number of cases, after the cap.
        fidelity: The integration fidelity.
        duration_s: Each case's simulated duration, in seconds.
        directory: The directory the test's run files are written to, in the active session.
        vehicle: The vehicle's name in `VEHICLES`, or None for the scenario's own.
    """

    scenario: str
    runs: int
    fidelity: str
    duration_s: float
    directory: Path
    vehicle: str | None = None


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--full-monte-carlo` and `--sim-session` options and the `sim_out` setting.

    Args:
        parser: The pytest command-line parser.
    """
    group = parser.getgroup("wise-dynamics", "WiseDynamics vehicle cases")
    group.addoption(
        "--full-monte-carlo",
        action="store_true",
        default=False,
        help=f"fly every case of every test; without it each test flies at most {_CAPPED_RUNS}",
    )
    group.addoption(
        "--sim-session",
        nargs="?",
        const="",
        default=None,
        metavar="name",
        help="with a name, make it the active Monte Carlo session; without, list the sessions; "
        "either way no test runs",
    )
    parser.addini(
        "sim_out",
        type="string",
        default=_DEFAULT_OUT,
        help="directory that holds the Monte Carlo sessions and the `SESSION` pointer, relative "
        f"to the rootdir; default {_DEFAULT_OUT}",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> int | None:
    """Handle `--sim-session` without configuring the session or running any test.

    The handler reads only the command line and the `sim_out` ini setting, so a broken pointer or
    an install that cannot be described never blocks it. When `WISE_DYNAMICS_SESSION_REPORT` names
    a file, the list mode writes the session directory `session_directory` gives, or null when no
    session is active, to it as JSON.

    Args:
        config: The parsed, not yet configured, pytest configuration.

    Returns:
        The exit code when `--sim-session` was given, else None. An invalid name or an empty
        `sim_out` ini setting stops the run with a usage error.
    """
    name = config.getoption("sim_session")
    if name is None:
        return None
    root = sessions.sim_root(config)
    report = os.environ.get(sessions.REPORT_ENV)
    if report and not name:
        try:
            found = str(sessions.session_directory(config))
        except sessions.NoSessionError:
            found = None
        Path(report).write_text(json.dumps(found))
    if name:
        sessions.check_name(name, "--sim-session")
        sessions.write_pointer(root, name)
        print(f"session: {name}")
        return int(pytest.ExitCode.OK)
    active = sessions.read_pointer(root)
    print(f"active: {active or 'none'}")
    others = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    for other in others:
        if other != active:
            print(other)
    return int(pytest.ExitCode.OK)


def pytest_configure(config: pytest.Config) -> None:
    """Register the `sim_` marks and check the `sim_out` ini setting.

    The active session is resolved when a test first needs it, not here.

    A usage error stops the pytest run when the `sim_out` ini setting is empty or the
    `wise-dynamics` install is neither a git pin nor an editable local checkout.

    Args:
        config: The session's pytest configuration, which holds the stash.
    """
    for name, argument, description in _MARKS:
        config.addinivalue_line("markers", f"{name}({argument}): {description}")
    sessions.sim_root(config)
    config.stash[_WRITTEN_DIRS] = set()
    config.stash[_PROVENANCE] = _provenance()
    config.stash[_CALLER] = _caller_provenance(config.rootpath)


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """Hand a starting pytest-xdist worker the active session, or why there is none.

    Args:
        node: The pytest-xdist worker node being configured.
    """
    try:
        handed = {"directory": str(_active_session(node.config)), "error": None}
    except pytest.UsageError as error:
        handed = {"directory": None, "error": str(error)}
    node.workerinput[_WORKER_SESSION] = handed


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: object) -> None:
    """Collect the directories a pytest-xdist worker wrote run files to.

    Args:
        node: The pytest-xdist worker node that finished.
        error: The error that ended the worker, if any.
    """
    workeroutput = getattr(node, "workeroutput", None) or {}
    node.config.stash[_WRITTEN_DIRS].update(
        Path(path) for path in workeroutput.get(_WORKER_DIRS, ())
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize every test that uses `sim_case_index` over its case numbers.

    With no usable active session the test gets the single case `no-session`, which errors at
    setup with the reason; other tests are unaffected.

    Args:
        metafunc: The test function being collected.
    """
    if "sim_case_index" not in metafunc.fixturenames:
        return
    node = metafunc.definition
    try:
        settings = _settings(node)
    except ValueError as error:
        pytest.fail(f"{node.nodeid}: {error}", pytrace=False)
    except pytest.UsageError as error:
        metafunc.parametrize(
            "sim_case_index", [pytest.param(_NoSession(str(error)), id="no-session")], indirect=True
        )
        return
    metafunc.parametrize("sim_case_index", range(settings.runs), indirect=True)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Write each collected test's planned case count and node path into the session's folder.

    The count is the test's `sim_runs` after the cap, not the number of cases selected. The run
    files and crash records the test's folder holds for cases from that count up are deleted.

    A `--collect-only` run writes nothing, and neither does a pytest-xdist controller, which
    collects nothing; each worker writes the same content. A test with no active Monte Carlo
    session has no folder to write to.

    Args:
        session: The finished collection session.
    """
    if session.config.option.collectonly:
        return
    planned: dict[Path, tuple[int, str]] = {}
    for item in session.items:
        if "sim_case_index" in item.fixturenames and _case_of(item) is not None:
            settings = _settings(item)
            planned[settings.directory] = (settings.runs, item.nodeid.partition("[")[0])
    for directory, (runs, nodeid) in planned.items():
        sessions.write_planned(directory, runs, nodeid)
        _delete_leftovers(directory, runs)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Delete a Monte Carlo case's earlier run file and crash record before it runs.

    Args:
        item: The test item about to be set up.
    """
    case = _case_of(item)
    if case is not None:
        directory = _run_dir(item)
        sessions.run_path(directory, case).unlink(missing_ok=True)
        sessions.crash_path(directory, case).unlink(missing_ok=True)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Any:
    """Record a crash for a Monte Carlo case that failed in setup or call without a run file.

    Args:
        item: The test item that ran.
        call: The setup, call or teardown phase that finished.

    Returns:
        The phase's report, unchanged.
    """
    report = yield
    case = _case_of(item)
    if (
        case is not None
        and report.when in ("setup", "call")
        and report.failed
        and call.excinfo is not None
    ):
        directory = _run_dir(item)
        if not sessions.run_path(directory, case).is_file():
            _write_crash(directory, case, call.excinfo.exconly(), str(report.longrepr))
    return report


@pytest.hookimpl(optionalhook=True)
def pytest_handlecrashitem(crashitem: str, report: pytest.TestReport, sched: Any) -> None:
    """Record a crash for the Monte Carlo case a pytest-xdist worker died in.

    Args:
        crashitem: The node id of the test the worker was running.
        report: The failure report pytest-xdist made for it.
        sched: The pytest-xdist scheduler, which holds the configuration.
    """
    located = _locate(crashitem)
    if located is None or _SESSION not in sched.config.stash:
        return
    directory, case = located
    directory = sched.config.stash[_SESSION] / directory
    if not (directory / sessions.PLANNED).is_file():
        return
    sessions.run_path(directory, case).unlink(missing_ok=True)
    crash = str(report.longrepr)
    _write_crash(directory, case, (crash.splitlines() or [""])[0], crash)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Report a pytest-xdist worker's written directories to the controller.

    Args:
        session: The finishing session, whose config holds the stash and the worker output.
    """
    workeroutput = getattr(session.config, "workeroutput", None)
    if workeroutput is not None:
        written = session.config.stash[_WRITTEN_DIRS]
        workeroutput[_WORKER_DIRS] = sorted(str(path) for path in written)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    """Report where the run files are, one line per directory that received any.

    Args:
        terminalreporter: pytest's terminal reporter, which writes the summary lines.
        config: The session's pytest configuration.
    """
    if hasattr(config, "workerinput"):
        return
    for directory in sorted(config.stash[_WRITTEN_DIRS]):
        count = len(sessions.read_test_results(directory).runs)
        terminalreporter.write_line(f"wise-dynamics: {_run_files(count)} in {directory}")


@pytest.fixture
def sim_case_index(request: pytest.FixtureRequest) -> int:
    """This case's number.

    A test using it, directly or through another `sim_` fixture, runs once per case.

    Args:
        request: The fixture request, whose `param` is the case number.

    Returns:
        The case number, from 0 up to the test's `sim_runs` (after the cap).
    """
    if isinstance(request.param, _NoSession):
        pytest.fail(request.param.reason, pytrace=False)
    return request.param


@pytest.fixture
def sim_duration_s(request: pytest.FixtureRequest) -> float:
    """Each case's simulated duration.

    Args:
        request: The fixture request, whose node carries the `sim_duration_s` mark.

    Returns:
        The test's `sim_duration_s`, in seconds.
    """
    return _settings(request.node).duration_s


@pytest.fixture
def sim_test_directory(request: pytest.FixtureRequest) -> Path:
    """The test's directory in the session its runs are written to.

    Args:
        request: The fixture request, whose config resolves the session.

    Returns:
        The directory `sim_run_file` writes this test's run files in: `scratch` for a capped run,
        else the active session.
    """
    return _settings(request.node).directory


@pytest.fixture
def sim_scenario(request: pytest.FixtureRequest, sim_case_index: int) -> "Scenario":
    """This case's scenario, dispersed from the case's two random streams.

    Args:
        request: The fixture request, whose node carries the `sim_scenario` and `sim_fidelity`
            marks.
        sim_case_index: The case number.

    Returns:
        The test's `sim_scenario` at its `sim_fidelity`, dispersed from the case's first stream,
        with its vehicle's sensor noise seeded from the case's second. Its vehicle is
        `sim_scenario.vehicle`.
    """
    settings = _settings(request.node)
    draws_stream, noise_stream = _streams(sim_case_index)
    scenario_type = _scenario_type(settings.scenario)
    if settings.vehicle is None:
        scenario = scenario_type()
    else:
        scenario = scenario_type(_vehicle_type(settings.vehicle))
    scenario.fidelity = settings.fidelity
    scenario.disperse(
        np.random.default_rng(draws_stream), noise_seed=int(noise_stream.generate_state(1)[0])
    )
    return scenario


@pytest.fixture
def sim_run_file(
    request: pytest.FixtureRequest,
    sim_case_index: int,
    sim_scenario: "Scenario",
) -> Callable[..., Path]:
    """A writer of this case's run file, in the test's directory of the active session.

    The file is `run_<NNNN>.h5`, written through `sim_scenario.vehicle` and laid out as the basilisk
    README describes. The file's root attributes are `case`, `seed`, `fidelity`, `created_utc` (the
    case's start, ISO 8601 UTC), `versions_bsk`, `versions_wise_dynamics`, the install's provenance
    (`versions_wise_dynamics_*`, see the basilisk README), the rootdir repository's commit and
    dirty state (`versions_caller_commit` and `versions_caller_dirty`, absent when the rootdir is
    not a git checkout) and one `draw_<name>` per value the scenario drew.

    Args:
        request: The fixture request, whose node carries the marks and whose config stash records
            the directories written to.
        sim_case_index: The case number.
        sim_scenario: The case's scenario, whose vehicle writes the file.

    Returns:
        `sim_run_file(datasets, attrs=None)`, which writes the file and returns its path.
    """
    own: dict[str, Any] = {
        "case": sim_case_index,
        "seed": _SEED,
        "fidelity": sim_scenario.fidelity,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions_wise_dynamics": version("wise-dynamics"),
        "versions_bsk": version("bsk"),
        **{
            f"versions_wise_dynamics_{name}": value
            for name, value in request.config.stash[_PROVENANCE].items()
        },
        **{
            f"versions_caller_{name}": value
            for name, value in request.config.stash[_CALLER].items()
        },
    }
    for name, value in sim_scenario.draws.items():
        own[f"draw_{name}"] = np.asarray(value, dtype=float)
    directory = _settings(request.node).directory
    path = sessions.run_path(directory, sim_case_index)
    config = request.config

    def write(
        datasets: Mapping[str, tuple[Any, str]], attrs: Mapping[str, Any] | None = None
    ) -> Path:
        """Write the run file.

        Args:
            datasets: Dataset name to `(array, units)`, as `Scenario.run` returns them.
            attrs: Root attributes written beside the file's own.

        Returns:
            The file's path.

        Raises:
            ValueError: `attrs` names one of the file's own root attributes.
        """
        extras = dict(attrs or {})
        replaced = sorted(set(own) & set(extras))
        if replaced:
            raise ValueError(f"attrs may not replace the run file's own attrs {replaced}")
        sim_scenario.vehicle.write_run_file(path, datasets, {**own, **extras})
        config.stash[_WRITTEN_DIRS].add(directory)
        return path

    return write


def _settings(node: pytest.Function) -> _Settings:
    """Resolve a test's configuration from its marks, the defaults and the cap.

    Resolving the run directory raises `pytest.UsageError` through `_active_session` when no
    session is active or its name is invalid.

    Args:
        node: The test function, or one of its parametrized items.

    Returns:
        The test's settings, its case count capped at two unless `--full-monte-carlo` is given.

    Raises:
        ValueError: A mark's value is malformed; the message starts with the mark's name.
    """
    values = _marked_values(node)
    scenario = values.get("sim_scenario", _DEFAULT_SCENARIO)
    runs = values.get("sim_runs", _DEFAULT_RUNS)
    fidelity = values.get("sim_fidelity", _DEFAULT_FIDELITY)
    duration_s = values.get("sim_duration_s", _DEFAULT_DURATION_S)
    if not isinstance(scenario, str):
        raise ValueError(f"sim_scenario: expected a scenario name, got {scenario!r}")
    try:
        _scenario_type(scenario)
    except ValueError as error:
        raise ValueError(f"sim_scenario: {error}") from None
    if not isinstance(runs, numbers.Integral) or isinstance(runs, bool) or runs < 1:
        raise ValueError(f"sim_runs: expected an integer of at least 1, got {runs!r}")
    if fidelity not in _FIDELITIES:
        raise ValueError(f"sim_fidelity: expected one of {_FIDELITIES}, got {fidelity!r}")
    if (
        not isinstance(duration_s, numbers.Real)
        or isinstance(duration_s, bool)
        or not duration_s > 0
    ):
        raise ValueError(f"sim_duration_s: expected a positive number, got {duration_s!r}")
    vehicle = values.get("sim_vehicle")
    if vehicle is not None:
        try:
            _vehicle_type(vehicle)
        except ValueError as error:
            raise ValueError(f"sim_vehicle: {error}") from None
    if _capped(node.config):
        runs = min(runs, _CAPPED_RUNS)
    directory = _run_dir(node)
    return _Settings(scenario, int(runs), fidelity, float(duration_s), directory, vehicle)


def _run_dir(node: pytest.Function) -> Path:
    """Build a test's run directory in the active session.

    `_active_session` raises `pytest.UsageError` when no session is active or its name is invalid.

    Args:
        node: The test function, or one of its parametrized items.

    Returns:
        `_test_directory` under the active session's directory.
    """
    return _test_directory(node, _active_session(node.config))


def _active_session(config: pytest.Config) -> Path:
    """Resolve the run's session directory once and cache it in the stash.

    A pytest-xdist worker uses the session its controller handed it.

    Args:
        config: The session's pytest configuration, which holds the stash.

    Returns:
        The `scratch` session's directory when the run is capped, else the active session's.

    Raises:
        pytest.UsageError: A full run has no active session, or its name is invalid.
    """
    if _SESSION not in config.stash:
        workerinput = getattr(config, "workerinput", None)
        if workerinput is None:
            config.stash[_SESSION] = (
                sessions.sim_root(config) / sessions.SCRATCH
                if _capped(config)
                else sessions.session_directory(config)
            )
        else:
            handed = workerinput[_WORKER_SESSION]
            if handed["directory"] is None:
                raise pytest.UsageError(handed["error"])
            config.stash[_SESSION] = Path(handed["directory"])
    return config.stash[_SESSION]


def _delete_leftovers(directory: Path, runs: int) -> None:
    """Delete a test's run files and crash records of cases numbered `runs` or higher.

    Args:
        directory: The test's directory in the session.
        runs: The test's planned case count.
    """
    for path in directory.glob("run_*"):
        match = _CASE_FILE.fullmatch(path.name)
        if match and int(match[1]) >= runs:
            path.unlink(missing_ok=True)


def _capped(config: pytest.Config) -> bool:
    """Tell whether the run flies at most two cases per test, into the `scratch` session.

    Args:
        config: The session's pytest configuration.

    Returns:
        True unless `--full-monte-carlo` is given; True whatever it says while the
        `WISE_DYNAMICS_CAPPED` environment variable is non-empty.
    """
    return bool(os.environ.get(sessions.CAPPED_ENV)) or not config.getoption("full_monte_carlo")


def _case_of(item: pytest.Item) -> int | None:
    """Find the case number of a Monte Carlo item that has a session.

    Args:
        item: Any collected test item.

    Returns:
        The case number, or None for an item with no `sim_case_index` or no active session.
    """
    callspec = getattr(item, "callspec", None)
    case = callspec.params.get("sim_case_index") if callspec else None
    return case if isinstance(case, int) else None


def _locate(nodeid: str) -> tuple[Path, int] | None:
    """Find where a Monte Carlo case lives from its node id, without collecting anything.

    Args:
        nodeid: A test's node id, such as `tests/test_x.py::TestA::test_f[3]`.

    Returns:
        The test's directory relative to the session, as `_test_directory` builds it, and the case
        number; None when the node id does not end in a case number.
    """
    file, _, rest = nodeid.partition("::")
    *names, last = rest.split("::")
    function, _, parameter = last.partition("[")
    case = parameter.removesuffix("]")
    if not case.isdigit():
        return None
    return _directory_in_session(Path(file), names, function), int(case)


def _write_crash(directory: Path, case: int, exception: str, traceback: str) -> None:
    """Write a case's crash record.

    Args:
        directory: The test's directory in the session.
        case: The case number.
        exception: The exception, as one line.
        traceback: The failure's full report.
    """
    directory.mkdir(parents=True, exist_ok=True)
    sessions.crash_path(directory, case).write_text(
        json.dumps({"exception": exception, "traceback": traceback})
    )


def _marked_values(node: pytest.Function) -> dict[str, Any]:
    """Read each `sim_` mark's single argument from the closest mark of its name.

    Args:
        node: The test function, or one of its parametrized items.

    Returns:
        Mark name to argument, for the marks the test carries.

    Raises:
        ValueError: A mark does not take exactly one positional argument and no keywords.
    """
    values: dict[str, Any] = {}
    for name, _, _ in _MARKS:
        mark = node.get_closest_marker(name)
        if mark is None:
            continue
        if len(mark.args) != 1 or mark.kwargs:
            raise ValueError(f"{name}: takes exactly one positional argument and no keywords")
        values[name] = mark.args[0]
    return values


def _test_directory(node: pytest.Function, root: Path) -> Path:
    """Build the directory a test's run files are written to.

    Args:
        node: The test function, or one of its parametrized items.
        root: The directory the test's campaigns are rooted at.

    Returns:
        `<root>/<test file>/<class>/.../<function>`, the test file being its path relative to
        the rootdir without `.py`, or its stem when it is outside the rootdir.
    """
    try:
        file = node.path.relative_to(node.config.rootpath)
    except ValueError:
        file = node.path
    classes = [item.name for item in node.listchain() if isinstance(item, pytest.Class)]
    return root / _directory_in_session(file, classes, node.originalname)


def _directory_in_session(file: Path, classes: Sequence[str], function: str) -> Path:
    """Build a test's directory within a session; the one place that layout is defined.

    Args:
        file: The test file, relative to the rootdir, or absolute or starting `..` when outside it.
        classes: The enclosing classes, outermost first.
        function: The test function's name.

    Returns:
        `<test file>/<class>/.../<function>`, the test file being its path without `.py`, or its
        stem when it is outside the rootdir.
    """
    file_part = file.with_suffix("")
    if file_part.is_absolute() or ".." in file_part.parts:
        file_part = Path(file_part.name)
    return file_part / Path(*classes) / function


def _provenance() -> dict[str, str | bool]:
    """Describe how `wise-dynamics` was installed, from its PEP 610 `direct_url.json`.

    Returns:
        `install` (`pinned` or `local`), `commit`, `requested_revision` (empty for a local
        install) and `dirty` (always False for a pinned install).

    Raises:
        pytest.UsageError: The install is neither a git pin nor an editable local checkout, or
            `git` cannot describe the checkout.
    """
    text = distribution("wise-dynamics").read_text("direct_url.json")
    found = json.loads(text) if text else None
    info = found or {}
    vcs_info = info.get("vcs_info", {})
    if vcs_info.get("vcs") == "git" and "commit_id" in vcs_info:
        return {
            "install": "pinned",
            "commit": vcs_info["commit_id"],
            "requested_revision": vcs_info.get("requested_revision", ""),
            "dirty": False,
        }
    url = info.get("url", "")
    if not (info.get("dir_info", {}).get("editable") and url.startswith("file://")):
        raise pytest.UsageError(
            f"the install of wise-dynamics is not a git pin or an editable checkout: {found}\n"
            f"{_INSTALL_HINT}"
        )
    git = ["git", "-C", url2pathname(urlparse(url).path)]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        status = subprocess.run(
            [*git, "status", "--porcelain"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise pytest.UsageError(
            f"git could not describe the editable checkout {git[2]}: {error}\n{_INSTALL_HINT}"
        ) from None
    return {
        "install": "local",
        "commit": commit.stdout.strip(),
        "requested_revision": "",
        "dirty": bool(status.stdout.strip()),
    }


def _caller_provenance(rootdir: Path) -> dict[str, str | bool]:
    """Describe the git repository that holds the rootdir.

    Args:
        rootdir: The pytest rootdir.

    Returns:
        `commit` (HEAD) and `dirty` (whether `git status --porcelain` lists anything); empty when
        `git` cannot describe the rootdir.
    """
    git = ["git", "-C", str(rootdir)]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        )
        status = subprocess.run(
            [*git, "status", "--porcelain"], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return {}
    return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}


def _run_files(count: int) -> str:
    """Say `count` run files in words.

    Args:
        count: The number of run files.

    Returns:
        `"1 run file"` or `"<count> run files"`.
    """
    return f"{count} run file" if count == 1 else f"{count} run files"


def _scenario_type(name: str) -> "type[Scenario]":
    """Look up a scenario class by name, importing Basilisk on first use.

    Args:
        name: A name in `SCENARIOS`.

    Returns:
        The scenario class `name` registers.

    Raises:
        ValueError: `name` is not registered.
    """
    from wise_dynamics.basilisk.scenarios import SCENARIOS

    try:
        return SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f"unknown scenario {name!r}; known scenarios: {', '.join(sorted(SCENARIOS))}"
        ) from None


def _vehicle_type(name: object) -> type:
    """Look up a vehicle class by name, importing Basilisk on first use.

    Args:
        name: A name in `VEHICLES`.

    Returns:
        The vehicle class `name` registers.

    Raises:
        ValueError: `name` is not registered.
    """
    from wise_dynamics.basilisk.vehicles import VEHICLES

    if name not in VEHICLES:
        raise ValueError(f"unknown vehicle {name!r}; known vehicles: {', '.join(sorted(VEHICLES))}")
    return VEHICLES[name]


def _streams(index: int) -> list[np.random.SeedSequence]:
    """Spawn case `index`'s two random streams from the seed `_SEED`.

    Args:
        index: The case number.

    Returns:
        The first stream, which draws the dispersions, and the second, which seeds sensor noise.
    """
    return np.random.SeedSequence(_SEED, spawn_key=(index,)).spawn(2)
