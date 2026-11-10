"""The WiseDynamics pytest plugin: each test that uses a `sim_` fixture is one Monte Carlo campaign.

A test configures its campaign with marks on the function, a class or the module, the closest
winning. Installing `wise-dynamics` registers this module through its `pytest11` entry point.
`wise_dynamics/basilisk/README.md` documents the marks, `--ci` and fixtures.
"""

import json
import numbers
import subprocess
from collections.abc import Callable, Mapping
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
_SEED = 0
_INSTALL_HINT = (
    "Install a pinned release with `uv sync` against a git dependency, or a local checkout with\n"
    "`uv pip install -e <checkout>`."
)
_DEFAULT_OUT = "monte_carlo"
_CI_RUNS = 2
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
    ("sim_runs", "n", f"number of cases, at least 1; default {_DEFAULT_RUNS}"),
    (
        "sim_fidelity",
        "level",
        "integration fidelity: hi is RK4 at 0.02 s, med RK4 at 0.05 s, lo RK2 at 0.05 s; "
        f"default {_DEFAULT_FIDELITY}",
    ),
    (
        "sim_duration_s",
        "seconds",
        f"simulated duration of each case, positive; default {_DEFAULT_DURATION_S:g}",
    ),
)


class _Settings(NamedTuple):
    """One test's resolved configuration.

    Attributes:
        scenario: The scenario's name in `SCENARIOS`.
        runs: The number of cases, after the `--ci` cap.
        fidelity: The integration fidelity.
        duration_s: Each case's simulated duration, in seconds.
        directory: The directory the test's run files are written to, in the active session.
    """

    scenario: str
    runs: int
    fidelity: str
    duration_s: float
    directory: Path


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--ci` option.

    Args:
        parser: The pytest command-line parser.
    """
    group = parser.getgroup("wise-dynamics", "WiseDynamics vehicle cases")
    group.addoption(
        "--ci",
        action="store_true",
        default=False,
        help=f"cap every test's case count at {_CI_RUNS}",
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
    an install that cannot be described never blocks it.

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
    if name:
        sessions.check_name(name, "--sim-session")
        root.mkdir(parents=True, exist_ok=True)
        (root / sessions.POINTER).write_text(f"{name}\n")
        print(f"session: {name}")
        return int(pytest.ExitCode.OK)
    pointer = root / sessions.POINTER
    active = pointer.read_text().strip() if pointer.is_file() else ""
    print(f"active: {active or 'none'}")
    others = sorted(path.name for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    for other in others:
        if other != active:
            print(other)
    return int(pytest.ExitCode.OK)


def pytest_configure(config: pytest.Config) -> None:
    """Register the `sim_` marks and check the `sim_out` ini setting.

    The active session is resolved when a test first needs it, not here.

    A usage error stops the session when the `sim_out` ini setting is empty or the `wise-dynamics`
    install is neither a git pin nor an editable local checkout.

    Args:
        config: The session's pytest configuration, which holds the stash.
    """
    for name, argument, description in _MARKS:
        config.addinivalue_line("markers", f"{name}({argument}): {description}")
    sessions.sim_root(config)
    config.stash[_WRITTEN_DIRS] = set()
    config.stash[_PROVENANCE] = _provenance()


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
            "sim_case_index", [pytest.param(str(error), id="no-session")], indirect=True
        )
        return
    metafunc.parametrize("sim_case_index", range(settings.runs), indirect=True)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Write each collected test's planned case count into the session's folder.

    A `--collect-only` session writes nothing, and neither does a pytest-xdist controller, which
    collects nothing; each worker writes the same content. A test with no active session has no
    folder to write to.

    Args:
        session: The finished collection session.
    """
    if session.config.option.collectonly:
        return
    planned: dict[Path, int] = {}
    for item in session.items:
        if "sim_case_index" in item.fixturenames and not _is_no_session(item):
            directory = _settings(item).directory
            planned[directory] = planned.get(directory, 0) + 1
    for directory, runs in planned.items():
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "planned.json").write_text(json.dumps({"runs": runs}))


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
        count = len(list(directory.glob("run_*.h5")))
        terminalreporter.write_line(f"wise-dynamics: {_run_files(count)} in {directory}")


@pytest.fixture
def sim_case_index(request: pytest.FixtureRequest) -> int:
    """This case's number.

    A test using it, directly or through another `sim_` fixture, runs once per case.

    Args:
        request: The fixture request, whose `param` is the case number.

    Returns:
        The case number, from 0 up to the test's `sim_runs` (after the `--ci` cap).
    """
    if isinstance(request.param, str):
        pytest.fail(request.param, pytrace=False)
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
    scenario = _scenario_type(settings.scenario)()
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
    """A writer of this case's run file, `<session folder>/run_<NNNN>.h5`.

    The file is written through `sim_scenario.vehicle`, in the session's folder under the test's
    directory, laid out as the basilisk README describes. The file's root attributes are `case`,
    `seed`, `fidelity`, `created_utc` (the case's start, ISO 8601 UTC), `versions_bsk`,
    `versions_wise_dynamics`, the install's provenance (`versions_wise_dynamics_*`, see the basilisk
    README) and one `draw_<name>` per value the scenario drew.

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
    }
    for name, value in sim_scenario.draws.items():
        own[f"draw_{name}"] = np.asarray(value, dtype=float)
    directory = _settings(request.node).directory
    path = directory / f"run_{sim_case_index:04d}.h5"
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
    """Resolve a test's configuration from its marks, the defaults and `--ci`.

    Args:
        node: The test function, or one of its parametrized items.

    Returns:
        The test's settings, its case count capped at two under `--ci`.

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
    if node.config.getoption("ci"):
        runs = min(runs, _CI_RUNS)
    directory = _run_dir(node)
    return _Settings(scenario, int(runs), fidelity, float(duration_s), directory)


def _run_dir(node: pytest.Function) -> Path:
    """Build a test's run directory in the active session.

    Args:
        node: The test function, or one of its parametrized items.

    Returns:
        `_test_directory` under the active session's directory.
    """
    return _test_directory(node, _active_session(node.config))


def _active_session(config: pytest.Config) -> Path:
    """Resolve the run's session directory once, so a later switch of the pointer cannot split it.

    A pytest-xdist worker uses the session its controller handed it.

    Args:
        config: The session's pytest configuration, which holds the stash.

    Returns:
        The active session's directory.

    Raises:
        pytest.UsageError: There is no active session, or its name is invalid.
    """
    if _SESSION not in config.stash:
        workerinput = getattr(config, "workerinput", None)
        if workerinput is None:
            config.stash[_SESSION] = sessions.session_directory(config)
        else:
            handed = workerinput[_WORKER_SESSION]
            if handed["directory"] is None:
                raise pytest.UsageError(handed["error"])
            config.stash[_SESSION] = Path(handed["directory"])
    return config.stash[_SESSION]


def _is_no_session(item: pytest.Function) -> bool:
    """Tell whether an item is the `no-session` case of a Monte Carlo test.

    Args:
        item: A collected test item that uses `sim_case_index`.

    Returns:
        Whether its case parameter is the reason text rather than a case number.
    """
    return isinstance(item.callspec.params["sim_case_index"], str)


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
        file_part = node.path.relative_to(node.config.rootpath).with_suffix("")
    except ValueError:
        # A test file outside the rootdir has no relative path; its stem stands in.
        file_part = Path(node.path.stem)
    classes = [item.name for item in node.listchain() if isinstance(item, pytest.Class)]
    return root / file_part / Path(*classes) / node.originalname


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


def _streams(index: int) -> list[np.random.SeedSequence]:
    """Spawn case `index`'s two random streams from the seed `_SEED`.

    Args:
        index: The case number.

    Returns:
        The first stream, which draws the dispersions, and the second, which seeds sensor noise.
    """
    return np.random.SeedSequence(_SEED, spawn_key=(index,)).spawn(2)
