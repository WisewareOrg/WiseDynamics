"""The WiseDynamics pytest plugin: a pytest session over `--sim-runs` cases is a Monte Carlo
campaign of one plant.

Installing `wise-dynamics` registers this module through its `pytest11` entry point.
`wise_dynamics/basilisk/README.md` documents its options and fixtures.
"""

import secrets
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pytest

from wise_dynamics.dispersion import apply_draws
from wise_dynamics.provenance import distribution_version
from wise_dynamics.run_file import Dataset, write_run

if TYPE_CHECKING:
    from wise_dynamics.basilisk.plant import Plant

__all__ = ["RunWriter", "sim_out_dir"]

_WORKER_OUT_DIR = "wise_dynamics_out_dir"


class RunWriter(Protocol):
    """The `sim_run_writer` fixture's value: writes the case's run file."""

    def __call__(
        self, datasets: Mapping[str, Dataset], extra_attrs: Mapping[str, Any] | None = None
    ) -> Path:
        """Write `<out_dir>/run_<NNNN>.h5` and return its path.

        Args:
            datasets: Dataset name to `Dataset`, as `wise_dynamics.write_run` takes them.
            extra_attrs: Further root attributes, for example the flight software's version as
                `versions_<distribution>`.

        Raises:
            ValueError: An entry of `extra_attrs` names one of the case's own root attributes.
            FileExistsError: The run file already exists.
        """
        ...


@dataclass(frozen=True)
class _Options:
    """The session's `--sim-*` options, parsed and checked once."""

    plant: str
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


_OPTIONS = pytest.StashKey[_Options]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--sim-*` options."""
    group = parser.getgroup("wise-dynamics", "WiseDynamics Monte Carlo campaign")
    group.addoption(
        "--sim-runs", type=int, default=8, help="number of cases in the campaign (default: 8)"
    )
    group.addoption(
        "--sim-out",
        default=None,
        help="directory the run files are written to (default: a new directory under the "
        "system temporary directory)",
    )
    group.addoption("--sim-seed", type=int, default=0, help="campaign seed (default: 0)")
    group.addoption(
        "--sim-duration-s",
        type=float,
        default=5700.0,
        help="simulated duration of each case, in seconds (default: 5700)",
    )
    group.addoption(
        "--sim-spacecraft",
        default="imu_spacecraft",
        help="name of the plant in wise_dynamics.basilisk.PLANTS (default: imu_spacecraft)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Parse and check the `--sim-*` options once, for every fixture of the session.

    The controlling process resolves the output directory to one absolute path, from
    `--sim-out` or a fresh name under the system temporary directory, and every pytest-xdist
    worker receives that path. The first run file written creates the directory.

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
                f"--sim-out {out_dir} already holds {len(stale)} run files; "
                "give each campaign a directory of its own"
            )
    else:
        out_dir = Path(tempfile.gettempdir()) / f"wise_dynamics_{secrets.token_hex(8)}"

    try:
        options = _Options(
            plant=config.getoption("--sim-spacecraft"),
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
    node.workerinput[_WORKER_OUT_DIR] = str(sim_out_dir(node.config))


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize every test that uses `sim_case_index` over the case numbers.

    Raises:
        pytest.UsageError: `--sim-spacecraft` names no plant in `PLANTS`.
    """
    if "sim_case_index" not in metafunc.fixturenames:
        return
    options = metafunc.config.stash[_OPTIONS]
    try:
        _plant_type(options.plant)
    except ValueError as error:
        raise pytest.UsageError(f"wise-dynamics: --sim-spacecraft: {error}") from None
    metafunc.parametrize("sim_case_index", range(options.runs), indirect=True)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    """Report where the run files are, when the session wrote any."""
    if hasattr(config, "workerinput"):
        return
    out_dir = sim_out_dir(config)
    if out_dir.is_dir():
        count = len(list(out_dir.glob("run_*.h5")))
        terminalreporter.write_line(f"wise-dynamics: {count} run files in {out_dir}")


def sim_out_dir(config: pytest.Config) -> Path:
    """The directory this session's run files go to, for a consumer's hooks such as a
    `pytest_sessionfinish` that checks the run files.

    Args:
        config: The session's pytest config, on the controller or a worker.

    Returns:
        The absolute directory, the same on the controller and every worker. It exists once a
        run file has been written.
    """
    return config.stash[_OPTIONS].out_dir


@pytest.fixture
def sim_case_index(request: pytest.FixtureRequest) -> int:
    """This case's number, `0 <= index < --sim-runs`. A test using it, directly or through
    `sim_draws`, `sim_plant` or `sim_run_writer`, runs once per case."""
    return request.param


@pytest.fixture
def sim_draws(request: pytest.FixtureRequest, sim_case_index: int) -> Mapping[str, np.ndarray]:
    """This case's draws: configuration field name to the value its plant is built with and its
    run file records.

    The default is the plant's standard dispersion, drawn from the case's own stream. A scenario
    supplies its own draws by overriding this fixture.
    """
    options = request.config.stash[_OPTIONS]
    plant_type = _plant_type(options.plant)
    draws_stream, _ = _streams(options.seed, sim_case_index)
    rng = np.random.default_rng(draws_stream)
    return plant_type.standard_dispersion.draw(plant_type.nominal(), rng)


@pytest.fixture
def sim_plant(
    request: pytest.FixtureRequest, sim_case_index: int, sim_draws: Mapping[str, Any]
) -> "Plant[Any]":
    """This case's plant: the nominal configuration with `sim_draws` applied, its sensor noise
    seeded from the case's own stream."""
    options = request.config.stash[_OPTIONS]
    plant_type = _plant_type(options.plant)
    config = apply_draws(plant_type.nominal(), sim_draws)
    return plant_type(config, noise_seed=_noise_seed(options.seed, sim_case_index))


@pytest.fixture
def sim_run_writer(
    request: pytest.FixtureRequest, sim_case_index: int, sim_draws: Mapping[str, Any]
) -> RunWriter:
    """Writes this case's run file, `<out_dir>/run_<NNNN>.h5`: `writer(datasets,
    extra_attrs=None) -> Path`.

    The file carries the root attributes `case`, `seed`, `created_utc`, `producer`,
    `versions_bsk`, `versions_wise_dynamics` and one `draw_<field>` per entry of `sim_draws`.
    """
    options = request.config.stash[_OPTIONS]

    def write(
        datasets: Mapping[str, Dataset], extra_attrs: Mapping[str, Any] | None = None
    ) -> Path:
        attrs = _root_attrs(options, sim_case_index, sim_draws)
        extra_attrs = extra_attrs or {}
        reserved = sorted(set(attrs) & set(extra_attrs))
        if reserved:
            raise ValueError(f"extra_attrs may not set the reserved root attrs {reserved}")
        path = options.out_dir / f"run_{sim_case_index:04d}.h5"
        return write_run(path, datasets, {**attrs, **extra_attrs})

    return write


def _plant_type(name: str) -> "type[Plant[Any]]":
    """The plant class `name` registers in `PLANTS`, imported with Basilisk on first use."""
    from wise_dynamics.basilisk.plants import PLANTS

    try:
        return PLANTS[name]
    except KeyError:
        raise ValueError(
            f"unknown plant {name!r}; known plants: {', '.join(sorted(PLANTS))}"
        ) from None


def _streams(seed: int, index: int) -> list[np.random.SeedSequence]:
    """Case `index`'s two random streams: the first draws its configuration, the second seeds
    its sensor noise."""
    return np.random.SeedSequence(seed, spawn_key=(index,)).spawn(2)


def _noise_seed(seed: int, index: int) -> int:
    """Case `index`'s sensor-noise seed, from its second stream."""
    _, noise_stream = _streams(seed, index)
    return int(noise_stream.generate_state(1)[0])


def _root_attrs(options: _Options, index: int, draws: Mapping[str, Any]) -> dict[str, Any]:
    """The root attributes of case `index`'s run file."""
    attrs: dict[str, Any] = {
        "case": index,
        "seed": options.seed,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions_wise_dynamics": distribution_version("wise-dynamics"),
        **_plant_type(options.plant).provenance(),
    }
    for name, value in draws.items():
        attrs[f"draw_{name}"] = np.asarray(value, dtype=float)
    return attrs
