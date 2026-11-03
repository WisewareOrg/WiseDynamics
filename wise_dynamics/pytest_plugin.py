"""The WiseDynamics pytest plugin: a pytest session runs one plant over `--sim-runs` Monte Carlo
cases.

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

from wise_dynamics.dispersion import apply_draws
from wise_dynamics.run_file import write_run
from wise_dynamics.session_options import SessionOptions

if TYPE_CHECKING:
    from wise_dynamics.basilisk.plant import Plant

__all__: list[str] = []

_WORKER_OUT_DIR = "wise_dynamics_out_dir"
_OPTIONS = pytest.StashKey[SessionOptions]()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--sim-*` options."""
    group = parser.getgroup("wise-dynamics", "WiseDynamics plant cases")
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
                f"--sim-out {out_dir} already holds {_run_files(len(stale))}; "
                "give each session a directory of its own"
            )
    else:
        out_dir = Path(tempfile.gettempdir()) / f"wise_dynamics_{secrets.token_hex(8)}"

    try:
        options = SessionOptions(
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
    node.workerinput[_WORKER_OUT_DIR] = str(node.config.stash[_OPTIONS].out_dir)


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
) -> "Plant":
    """This case's plant: the nominal configuration with `sim_draws` applied, its sensor noise
    seeded from the case's own stream."""
    options = request.config.stash[_OPTIONS]
    plant_type = _plant_type(options.plant)
    config = apply_draws(plant_type.nominal(), sim_draws)
    return plant_type(config, noise_seed=_noise_seed(options.seed, sim_case_index))


@pytest.fixture
def sim_run_writer(
    request: pytest.FixtureRequest, sim_case_index: int, sim_draws: Mapping[str, Any]
) -> Callable[..., Path]:
    """Writes this case's run file, `<out_dir>/run_<NNNN>.h5`: `writer(datasets,
    extra_attrs=None) -> Path`, with `datasets` mapping a name to `(array, units)`.

    The file carries the root attributes `case`, `seed`, `created_utc`, `versions_wise_dynamics`,
    `versions_bsk` and one `draw_<field>` per entry of `sim_draws`. An `extra_attrs` entry naming
    one of these raises `ValueError`.
    """
    options = request.config.stash[_OPTIONS]

    def write(
        datasets: Mapping[str, tuple[Any, str]], extra_attrs: Mapping[str, Any] | None = None
    ) -> Path:
        attrs = _root_attrs(options, sim_case_index, sim_draws)
        extra_attrs = extra_attrs or {}
        reserved = sorted(set(attrs) & set(extra_attrs))
        if reserved:
            raise ValueError(f"extra_attrs may not set the reserved root attrs {reserved}")
        path = options.out_dir / f"run_{sim_case_index:04d}.h5"
        return write_run(path, datasets, {**attrs, **extra_attrs})

    return write


def _run_files(count: int) -> str:
    """`count` run files, in words."""
    return f"{count} run file" if count == 1 else f"{count} run files"


def _plant_type(name: str) -> "type[Plant]":
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


def _root_attrs(options: SessionOptions, index: int, draws: Mapping[str, Any]) -> dict[str, Any]:
    """The root attributes of case `index`'s run file."""
    attrs: dict[str, Any] = {
        "case": index,
        "seed": options.seed,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "versions_wise_dynamics": version("wise-dynamics"),
        "versions_bsk": version("bsk"),
    }
    for name, value in draws.items():
        attrs[f"draw_{name}"] = np.asarray(value, dtype=float)
    return attrs
