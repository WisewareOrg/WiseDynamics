"""The WiseDynamics pytest plugin: a pytest session runs one vehicle over `--sim-runs` Monte Carlo
cases.

Installing `wise-dynamics` registers this module through its `pytest11` entry point.
`wise_dynamics/basilisk/README.md` documents its options and fixtures.
"""

import secrets
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from wise_dynamics.run_file import RunFile

if TYPE_CHECKING:
    from wise_dynamics.basilisk.vehicle import Vehicle

__all__: list[str] = []

_WORKER_OUT_DIR = "wise_dynamics_out_dir"


@dataclass(frozen=True)
class SessionOptions:
    """A pytest session's `--sim-*` options, parsed and checked once by the plugin.

    Attributes:
        vehicle: `--sim-vehicle`, a name in `wise_dynamics.basilisk.VEHICLES`.
        seed: `--sim-seed`, non-negative.
        runs: `--sim-runs`, at least 1.
        duration_s: `--sim-duration-s`, positive.
        out_dir: The directory the run files go to.

    Raises:
        ValueError: `seed`, `runs` or `duration_s` is out of range.
    """

    vehicle: str
    seed: int
    runs: int
    duration_s: float
    out_dir: Path

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError(f"--sim-seed must be non-negative, got {self.seed}")
        if self.runs < 1:
            raise ValueError(f"--sim-runs must be at least 1, got {self.runs}")
        if not self.duration_s > 0:
            raise ValueError(f"--sim-duration-s must be positive, got {self.duration_s}")


_OPTIONS = pytest.StashKey[SessionOptions]()


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
        "--sim-vehicle",
        default="imu_spacecraft",
        help="name of the vehicle in wise_dynamics.basilisk.VEHICLES (default: imu_spacecraft)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Parse and check the `--sim-*` options once, for every fixture of the session.

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

    try:
        options = SessionOptions(
            vehicle=config.getoption("--sim-vehicle"),
            seed=config.getoption("--sim-seed"),
            runs=config.getoption("--sim-runs"),
            duration_s=config.getoption("--sim-duration-s"),
            out_dir=out_dir,
        )
    except ValueError as error:
        raise pytest.UsageError(f"wise-dynamics: {error}") from None
    config.stash[_OPTIONS] = options


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """pytest-xdist hook: hand a starting worker the controller's output directory."""
    node.workerinput[_WORKER_OUT_DIR] = str(node.config.stash[_OPTIONS].out_dir)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize every test that uses `sim_case_index` over the case numbers.

    Raises:
        pytest.UsageError: `--sim-vehicle` names no vehicle in `VEHICLES`.
    """
    if "sim_case_index" not in metafunc.fixturenames:
        return
    options = metafunc.config.stash[_OPTIONS]
    try:
        _vehicle_type(options.vehicle)
    except ValueError as error:
        raise pytest.UsageError(f"wise-dynamics: --sim-vehicle: {error}") from None
    metafunc.parametrize("sim_case_index", range(options.runs), indirect=True)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    """Report where the run files are, when the session wrote any."""
    if hasattr(config, "workerinput"):
        return
    out_dir = config.stash[_OPTIONS].out_dir
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
    return request.config.stash[_OPTIONS].duration_s


@pytest.fixture
def sim_draws(request: pytest.FixtureRequest, sim_case_index: int) -> Mapping[str, np.ndarray]:
    """This case's draws: vehicle keyword to the value its vehicle is built with and its run file
    records. The default draws each `standard_dispersion` entry, in order, as the keyword's
    default plus independent normal samples of that 1-sigma from the case's first stream; a
    scenario overrides this fixture to supply its own.
    """
    options = request.config.stash[_OPTIONS]
    vehicle_type = _vehicle_type(options.vehicle)
    draws_stream, _ = _streams(options.seed, sim_case_index)
    rng = np.random.default_rng(draws_stream)
    draws = {}
    for name, std in vehicle_type.standard_dispersion.items():
        nominal = np.asarray(getattr(vehicle_type, name), dtype=float)
        draws[name] = nominal + rng.normal(0.0, std, size=nominal.shape)
    return draws


@pytest.fixture
def sim_vehicle(
    request: pytest.FixtureRequest, sim_case_index: int, sim_draws: Mapping[str, Any]
) -> "Vehicle":
    """This case's vehicle, built with `sim_draws` as keywords and its sensor noise seeded from the
    case's second stream."""
    options = request.config.stash[_OPTIONS]
    vehicle_type = _vehicle_type(options.vehicle)
    return vehicle_type(**sim_draws, noise_seed=_noise_seed(options.seed, sim_case_index))


@pytest.fixture
def sim_run_file(
    request: pytest.FixtureRequest, sim_case_index: int, sim_draws: Mapping[str, Any]
) -> RunFile:
    """This case's `RunFile`, `<out_dir>/run_<NNNN>.h5`.

    Its own root attributes are `case`, `seed`, `created_utc` (the case's start, ISO 8601 UTC),
    `versions_bsk`, `versions_wise_dynamics` and one `draw_<keyword>` per entry of `sim_draws`.
    """
    options = request.config.stash[_OPTIONS]
    attrs: dict[str, Any] = {
        "case": sim_case_index,
        "seed": options.seed,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions_wise_dynamics": version("wise-dynamics"),
        "versions_bsk": version("bsk"),
    }
    for name, value in sim_draws.items():
        attrs[f"draw_{name}"] = np.asarray(value, dtype=float)
    return RunFile(options.out_dir / f"run_{sim_case_index:04d}.h5", attrs)


def _run_files(count: int) -> str:
    """`count` run files, in words."""
    return f"{count} run file" if count == 1 else f"{count} run files"


def _vehicle_type(name: str) -> "type[Vehicle]":
    """The vehicle class `name` registers in `VEHICLES`, imported with Basilisk on first use."""
    from wise_dynamics.basilisk.vehicles import VEHICLES

    try:
        return VEHICLES[name]
    except KeyError:
        raise ValueError(
            f"unknown vehicle {name!r}; known vehicles: {', '.join(sorted(VEHICLES))}"
        ) from None


def _streams(seed: int, index: int) -> list[np.random.SeedSequence]:
    """Case `index`'s two random streams: the first draws, the second seeds sensor noise."""
    return np.random.SeedSequence(seed, spawn_key=(index,)).spawn(2)


def _noise_seed(seed: int, index: int) -> int:
    """Case `index`'s sensor-noise seed."""
    _, noise_stream = _streams(seed, index)
    return int(noise_stream.generate_state(1)[0])
