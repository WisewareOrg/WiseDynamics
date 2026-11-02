"""The WiseDynamics pytest plugin: Monte Carlo campaign options and fixtures.

Installing `wise-dynamics` registers this module through its `pytest11` entry point.
`wise_dynamics/basilisk/README.md` documents its options, fixtures and accessor.
"""

import secrets
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pytest

from wise_dynamics.campaign import Campaign, Case
from wise_dynamics.run_file import Dataset

if TYPE_CHECKING:
    from wise_dynamics.basilisk.plant import Plant

__all__ = ["RunWriter", "campaign_from_config"]

_CAMPAIGN = pytest.StashKey[Campaign]()
_WORKER_OUT_DIR = "wise_dynamics_out_dir"


class RunWriter(Protocol):
    """The `sim_run_writer` fixture's value: writes the case's run file."""

    def __call__(
        self, datasets: Mapping[str, Dataset], extra_attrs: Mapping[str, Any] | None = None
    ) -> Path:
        """Write `<out_dir>/run_<NNNN>.h5` and return its path; see `wise_dynamics.Case.write`."""
        ...


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the `--sim-*` campaign options."""
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
    """Build the session's `Campaign` from the options and store it in `config.stash`.

    The controlling process resolves the output directory to one absolute path, from
    `--sim-out` or a fresh name under the system temporary directory, and every pytest-xdist
    worker receives that path. The directory itself is created by the first run file written.

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
        campaign = Campaign(
            plant=config.getoption("--sim-spacecraft"),
            seed=config.getoption("--sim-seed"),
            runs=config.getoption("--sim-runs"),
            duration_s=config.getoption("--sim-duration-s"),
            out_dir=out_dir,
        )
    except ValueError as error:
        raise pytest.UsageError(f"wise-dynamics: {error}") from None
    config.stash[_CAMPAIGN] = campaign


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """pytest-xdist hook: hand a starting worker the controller's output directory."""
    node.workerinput[_WORKER_OUT_DIR] = str(campaign_from_config(node.config).out_dir)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize every test that uses `sim_case_index` over the campaign's case numbers.

    Raises:
        pytest.UsageError: `--sim-spacecraft` names no known plant.
    """
    if "sim_case_index" not in metafunc.fixturenames:
        return
    campaign = campaign_from_config(metafunc.config)
    try:
        _ = campaign.plant_type
    except ValueError as error:
        raise pytest.UsageError(f"wise-dynamics: --sim-spacecraft: {error}") from None
    metafunc.parametrize("sim_case_index", range(campaign.runs), indirect=True)


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    """Report where the campaign's run files are, when the session wrote any."""
    if hasattr(config, "workerinput"):
        return
    out_dir = campaign_from_config(config).out_dir
    if out_dir.is_dir():
        count = len(list(out_dir.glob("run_*.h5")))
        terminalreporter.write_line(f"wise-dynamics: {count} run files in {out_dir}")


def campaign_from_config(config: pytest.Config) -> Campaign:
    """The session's `Campaign`, for a consumer's hooks, such as a session-finish check of the
    campaign directory.

    Args:
        config: The pytest config of the session, on the controller or a worker.

    Returns:
        The `Campaign` built from the `--sim-*` options, with its absolute `out_dir`.
    """
    return config.stash[_CAMPAIGN]


@pytest.fixture(scope="session")
def sim_campaign(request: pytest.FixtureRequest) -> Campaign:
    """The session's `Campaign`: plant name, seed, run count, case duration in seconds
    (`duration_s`) and output directory."""
    return campaign_from_config(request.config)


@pytest.fixture
def sim_case_index(request: pytest.FixtureRequest) -> int:
    """This case's number, `0 <= index < --sim-runs`; a test using it, directly or through
    another `sim_` fixture, runs once per case."""
    return request.param


@pytest.fixture
def sim_draws(sim_campaign: Campaign, sim_case_index: int) -> Mapping[str, np.ndarray]:
    """This case's draws: configuration field name to the value the plant is built with.

    The default is the plant's standard dispersion drawn from the case's own seed stream. A
    scenario supplies its own draws by overriding this fixture, drawing from
    `sim_campaign.draws_rng(sim_case_index)`.
    """
    return sim_campaign.standard_draws(sim_case_index)


@pytest.fixture
def sim_case(sim_campaign: Campaign, sim_case_index: int, sim_draws: Mapping[str, Any]) -> Case:
    """This case: its index, its `sim_draws`, its configuration and sensor-noise seed, its run
    path."""
    return sim_campaign.case(sim_case_index, sim_draws)


@pytest.fixture
def sim_plant(sim_case: Case) -> "Plant[Any]":
    """This case's plant, built from `sim_case`'s configuration and sensor-noise seed."""
    return sim_case.build_plant()


@pytest.fixture
def sim_run_writer(sim_case: Case) -> RunWriter:
    """Writes this case's run file: `writer(datasets, extra_attrs=None) -> Path`.

    The file is `<out_dir>/run_<NNNN>.h5` and carries the case's root attributes and
    `draw_<name>` attributes; see `wise_dynamics.Case.write`.
    """
    return sim_case.write
