"""The WiseDynamics pytest plugin: each test that uses a `sim_` fixture is one Monte Carlo campaign.

A test configures its campaign with marks on the function, a class or the module, the closest
winning: `sim_scenario`, `sim_runs`, `sim_fidelity`, `sim_duration_s` and `sim_out`. A test runs
once per case and writes its run files under its own directory, so campaigns in one session are
separate. `--ci` caps every campaign at two cases. Installing `wise-dynamics` registers this
module through its `pytest11` entry point. `wise_dynamics/basilisk/README.md` documents the marks
and fixtures.
"""

import numbers
import os
import secrets
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import pytest

if TYPE_CHECKING:
    from wise_dynamics.basilisk.scenario import Scenario

__all__: list[str] = []

_WORKER_OUT_DIR = "wise_dynamics_out_dir"
_WORKER_DIRS = "wise_dynamics_dirs"
_OUT_DIR = pytest.StashKey[Path]()
_WRITTEN_DIRS = pytest.StashKey[set[Path]]()
_SEED = 0
_CI_RUNS = 2
_MARKS = (
    (
        "sim_scenario",
        "name",
        "name of the scenario in wise_dynamics.basilisk.SCENARIOS; default leo500",
    ),
    ("sim_runs", "n", "number of cases, at least 1; default 8"),
    (
        "sim_fidelity",
        "level",
        "integration fidelity: hi is RK4 at 0.02 s, med RK4 at 0.05 s, lo RK2 at 0.05 s; "
        "default lo",
    ),
    ("sim_duration_s", "seconds", "simulated duration of each case, positive; default 5700"),
    (
        "sim_out",
        "path",
        "directory the run files are written under, relative to the rootdir; default a new "
        "directory under the system temporary directory",
    ),
)


class _Settings(NamedTuple):
    """One test's resolved configuration.

    Attributes:
        scenario: The scenario's name in `SCENARIOS`.
        runs: The number of cases, after the `--ci` cap.
        fidelity: The integration fidelity.
        duration_s: Each case's simulated duration, in seconds.
        directory: The directory the test's run files are written to.
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


def pytest_configure(config: pytest.Config) -> None:
    """Register the `sim_` marks and resolve the session's default output directory.

    The controlling process resolves one absolute default directory, which every pytest-xdist
    worker receives; the first run file written creates it.

    Args:
        config: The session's pytest configuration, which holds the stash.
    """
    for name, argument, description in _MARKS:
        config.addinivalue_line("markers", f"{name}({argument}): {description}")
    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        out_dir = Path(workerinput[_WORKER_OUT_DIR])
    else:
        out_dir = Path(tempfile.gettempdir()) / f"wise_dynamics_{secrets.token_hex(8)}"
    config.stash[_OUT_DIR] = out_dir
    config.stash[_WRITTEN_DIRS] = set()


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """Hand a starting pytest-xdist worker the controller's default output directory.

    Args:
        node: The pytest-xdist worker node being configured.
    """
    node.workerinput[_WORKER_OUT_DIR] = str(node.config.stash[_OUT_DIR])


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
    stale = sorted(settings.directory.glob("run_*.h5"))
    if stale:
        pytest.fail(
            f"{node.nodeid}: {settings.directory} already holds {_run_files(len(stale))}; "
            "give each campaign a directory of its own",
            pytrace=False,
        )
    metafunc.parametrize("sim_case_index", range(settings.runs), indirect=True)


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
        The case number, `0 <= index < ` the test's `sim_runs`.
    """
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
    """A writer of this case's run file, `<test directory>/run_<NNNN>.h5`.

    The file is written through `sim_scenario.vehicle`. The test directory is
    `<root>/<test file>/<class>/.../<function>`, where the root is the test's `sim_out`. The
    file's root attributes are `case`, `seed`, `fidelity`, `created_utc`
    (the case's start, ISO 8601 UTC), `versions_bsk`, `versions_wise_dynamics` and one
    `draw_<name>` per value the scenario drew.

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
        written = config.stash[_WRITTEN_DIRS]
        written.add(directory)
        if hasattr(config, "workeroutput"):
            config.workeroutput[_WORKER_DIRS] = sorted(str(path) for path in written)
        return path

    return write


def _settings(node: pytest.Item | pytest.Function) -> _Settings:
    """Resolve a test's configuration from its marks, the defaults and `--ci`.

    Args:
        node: The test function, or one of its parametrized items.

    Returns:
        The test's settings, its case count capped at two under `--ci`.

    Raises:
        ValueError: A mark's value is malformed; the message starts with the mark's name.
    """
    values: dict[str, Any] = {}
    for name, _, _ in _MARKS:
        mark = node.get_closest_marker(name)
        if mark is None:
            continue
        if len(mark.args) != 1 or mark.kwargs:
            raise ValueError(f"{name}: takes exactly one positional argument and no keywords")
        values[name] = mark.args[0]
    scenario = values.get("sim_scenario", "leo500")
    runs = values.get("sim_runs", 8)
    fidelity = values.get("sim_fidelity", "lo")
    duration_s = values.get("sim_duration_s", 5700.0)
    if not isinstance(scenario, str):
        raise ValueError(f"sim_scenario: expected a scenario name, got {scenario!r}")
    try:
        _scenario_type(scenario)
    except ValueError as error:
        raise ValueError(f"sim_scenario: {error}") from None
    if not isinstance(runs, numbers.Integral) or isinstance(runs, bool) or runs < 1:
        raise ValueError(f"sim_runs: expected an integer of at least 1, got {runs!r}")
    if fidelity not in ("hi", "med", "lo"):
        raise ValueError(f"sim_fidelity: expected 'hi', 'med' or 'lo', got {fidelity!r}")
    if (
        not isinstance(duration_s, numbers.Real)
        or isinstance(duration_s, bool)
        or not duration_s > 0
    ):
        raise ValueError(f"sim_duration_s: expected a positive number, got {duration_s!r}")
    config = node.config
    if "sim_out" in values:
        out = values["sim_out"]
        if not isinstance(out, str | os.PathLike):
            raise ValueError(f"sim_out: expected a path, got {out!r}")
        root = config.rootpath / out
    else:
        root = config.stash[_OUT_DIR]
    file_part = node.nodeid.split("::")[0]
    if Path(file_part).is_absolute() or file_part.startswith(".."):
        file_part = Path(file_part).stem
    classes = [item.name for item in node.listchain() if isinstance(item, pytest.Class)]
    directory = root / file_part.removesuffix(".py") / Path(*classes) / node.originalname
    if config.getoption("ci"):
        runs = min(runs, _CI_RUNS)
    return _Settings(scenario, int(runs), fidelity, float(duration_s), directory)


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
