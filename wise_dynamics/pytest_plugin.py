"""The WiseDynamics pytest plugin: a pytest session runs one vehicle in one scenario over
`--sim-runs` Monte Carlo cases.

Installing `wise-dynamics` registers this module through its `pytest11` entry point.
`wise_dynamics/basilisk/README.md` documents its options and fixtures.
"""

import secrets
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from wise_dynamics.basilisk.scenario import Scenario
    from wise_dynamics.basilisk.vehicle import Vehicle

__all__: list[str] = []

_WORKER_OUT_DIR = "wise_dynamics_out_dir"
_OUT_DIR = pytest.StashKey[Path]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--sim-*` options."""
    group = parser.getgroup("wise-dynamics", "WiseDynamics vehicle cases")
    group.addoption(
        "--sim-runs", type=int, default=8, help="number of cases in the session (default: 8)"
    )
    group.addoption(
        "--sim-out",
        default=None,
        help="directory the run files are written to (default: a new directory under the "
        "system temporary directory)",
    )
    group.addoption("--sim-seed", type=int, default=0, help="session seed (default: 0)")
    group.addoption(
        "--sim-duration-s",
        type=float,
        default=5700.0,
        help="simulated duration of each case, in seconds (default: 5700)",
    )
    group.addoption(
        "--sim-scenario",
        default="leo500",
        help="name of the scenario in wise_dynamics.basilisk.SCENARIOS (default: leo500)",
    )
    group.addoption(
        "--sim-vehicle",
        default="imu_spacecraft",
        help="name of the vehicle in wise_dynamics.basilisk.VEHICLES (default: imu_spacecraft)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Check the `--sim-*` options once and resolve the session's output directory.

    The controlling process resolves one absolute output directory, which every pytest-xdist
    worker receives; the first run file written creates it.

    Raises:
        pytest.UsageError: An option is out of range, or `--sim-out` already holds run files.
    """
    workerinput = getattr(config, "workerinput", None)
    if workerinput is not None:
        out_dir = Path(workerinput[_WORKER_OUT_DIR])
    elif config.getoption("--sim-out"):
        out_dir = Path(config.getoption("--sim-out")).absolute()
        stale = sorted(out_dir.glob("run_*.h5"))
        if stale:
            raise pytest.UsageError(
                f"--sim-out {out_dir} already holds {_run_files(len(stale))}; "
                "give each session a directory of its own"
            )
    else:
        out_dir = Path(tempfile.gettempdir()) / f"wise_dynamics_{secrets.token_hex(8)}"

    seed = config.getoption("--sim-seed")
    runs = config.getoption("--sim-runs")
    duration_s = config.getoption("--sim-duration-s")
    if seed < 0:
        raise pytest.UsageError(f"wise-dynamics: --sim-seed must be non-negative, got {seed}")
    if runs < 1:
        raise pytest.UsageError(f"wise-dynamics: --sim-runs must be at least 1, got {runs}")
    if not duration_s > 0:
        raise pytest.UsageError(
            f"wise-dynamics: --sim-duration-s must be positive, got {duration_s}"
        )
    config.stash[_OUT_DIR] = out_dir


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """pytest-xdist hook: hand a starting worker the controller's output directory."""
    node.workerinput[_WORKER_OUT_DIR] = str(node.config.stash[_OUT_DIR])


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize every test that uses `sim_case_index` over the case numbers.

    Raises:
        pytest.UsageError: `--sim-vehicle` or `--sim-scenario` names nothing in its registry.
    """
    if "sim_case_index" not in metafunc.fixturenames:
        return
    config = metafunc.config
    for option, lookup in (("--sim-vehicle", _vehicle_type), ("--sim-scenario", _scenario_type)):
        try:
            lookup(config.getoption(option))
        except ValueError as error:
            raise pytest.UsageError(f"wise-dynamics: {option}: {error}") from None
    metafunc.parametrize("sim_case_index", range(config.getoption("--sim-runs")), indirect=True)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    """Report where the run files are, when the session wrote any."""
    if hasattr(config, "workerinput"):
        return
    out_dir = config.stash[_OUT_DIR]
    if out_dir.is_dir():
        count = len(list(out_dir.glob("run_*.h5")))
        terminalreporter.write_line(f"wise-dynamics: {_run_files(count)} in {out_dir}")


@pytest.fixture
def sim_case_index(request: pytest.FixtureRequest) -> int:
    """This case's number, `0 <= index < --sim-runs`. A test using it, directly or through
    another per-case `sim_` fixture, runs once per case."""
    return request.param


@pytest.fixture
def sim_duration_s(request: pytest.FixtureRequest) -> float:
    """Each case's simulated duration, in seconds: `--sim-duration-s`."""
    return request.config.getoption("--sim-duration-s")


@pytest.fixture
def sim_draws(request: pytest.FixtureRequest, sim_case_index: int) -> Mapping[str, np.ndarray]:
    """This case's draws: scenario placement attribute or vehicle keyword to the value the case is
    built with and its run file records. The default is the scenario class's `draw`, then the
    vehicle class's, from the case's first stream; a consumer overrides this fixture to supply its
    own.
    """
    draws_stream, _ = _streams(request.config.getoption("--sim-seed"), sim_case_index)
    rng = np.random.default_rng(draws_stream)
    scenario_draws = _scenario_type(request.config.getoption("--sim-scenario")).draw(rng)
    vehicle_draws = _vehicle_type(request.config.getoption("--sim-vehicle")).draw(rng)
    return {**scenario_draws, **vehicle_draws}


@pytest.fixture
def sim_vehicle(
    request: pytest.FixtureRequest, sim_case_index: int, sim_draws: Mapping[str, Any]
) -> "Vehicle":
    """This case's vehicle, built with the entries of `sim_draws` outside its scenario's
    `standard_dispersion` as keywords, and its sensor noise seeded from the case's second
    stream."""
    _, noise_stream = _streams(request.config.getoption("--sim-seed"), sim_case_index)
    placement = _scenario_type(request.config.getoption("--sim-scenario")).standard_dispersion
    keywords = {name: value for name, value in sim_draws.items() if name not in placement}
    vehicle_type = _vehicle_type(request.config.getoption("--sim-vehicle"))
    return vehicle_type(**keywords, noise_seed=int(noise_stream.generate_state(1)[0]))


@pytest.fixture
def sim_scenario(
    request: pytest.FixtureRequest, sim_draws: Mapping[str, Any], sim_vehicle: "Vehicle"
) -> "Scenario":
    """This case's scenario: `--sim-scenario`, dispersed with the entries of `sim_draws` in its
    `standard_dispersion`, holding `sim_vehicle`."""
    scenario = _scenario_type(request.config.getoption("--sim-scenario"))()
    placement = scenario.standard_dispersion
    scenario.disperse({name: value for name, value in sim_draws.items() if name in placement})
    scenario.add(sim_vehicle)
    return scenario


@pytest.fixture
def sim_run_file(
    request: pytest.FixtureRequest,
    sim_case_index: int,
    sim_draws: Mapping[str, Any],
    sim_vehicle: "Vehicle",
) -> Callable[..., Path]:
    """Writes this case's run file, `<out_dir>/run_<NNNN>.h5`, through `sim_vehicle`.

    Called as `sim_run_file(datasets, attrs=None)`, with `datasets` as `Scenario.run` returns
    them, it returns the file's path. The file's root attributes are `case`, `seed`,
    `created_utc` (the case's start, ISO 8601 UTC), `versions_bsk`, `versions_wise_dynamics` and
    one `draw_<name>` per entry of `sim_draws`, with `attrs` beside them; `attrs` naming one of
    those raises `ValueError`.
    """
    own: dict[str, Any] = {
        "case": sim_case_index,
        "seed": request.config.getoption("--sim-seed"),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions_wise_dynamics": version("wise-dynamics"),
        "versions_bsk": version("bsk"),
    }
    for name, value in sim_draws.items():
        own[f"draw_{name}"] = np.asarray(value, dtype=float)
    path = request.config.stash[_OUT_DIR] / f"run_{sim_case_index:04d}.h5"

    def write(
        datasets: Mapping[str, tuple[Any, str]], attrs: Mapping[str, Any] | None = None
    ) -> Path:
        extras = dict(attrs or {})
        replaced = sorted(set(own) & set(extras))
        if replaced:
            raise ValueError(f"attrs may not replace the run file's own attrs {replaced}")
        sim_vehicle.write_run_file(path, datasets, {**own, **extras})
        return path

    return write


def _run_files(count: int) -> str:
    """`count` run files, in words."""
    return f"{count} run file" if count == 1 else f"{count} run files"


def _vehicle_type(name: str) -> "type[Vehicle]":
    """The vehicle class `name` registers in `VEHICLES`, imported with Basilisk on first use.

    Raises:
        ValueError: `name` is not registered.
    """
    from wise_dynamics.basilisk.vehicles import VEHICLES

    try:
        return VEHICLES[name]
    except KeyError:
        raise ValueError(
            f"unknown vehicle {name!r}; known vehicles: {', '.join(sorted(VEHICLES))}"
        ) from None


def _scenario_type(name: str) -> "type[Scenario]":
    """The scenario class `name` registers in `SCENARIOS`, imported with Basilisk on first use.

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


def _streams(seed: int, index: int) -> list[np.random.SeedSequence]:
    """Case `index`'s two random streams: the first draws, the second seeds sensor noise."""
    return np.random.SeedSequence(seed, spawn_key=(index,)).spawn(2)
