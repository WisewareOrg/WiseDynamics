"""A pytest plugin exposing Monte Carlo campaign options and fixtures over this repository's
plant models, registered via this package's own `pytest11` entry point: installing
`wise-dynamics` loads it. Listing it again in a consumer's own `pytest_plugins` makes pytest
raise "Plugin already registered under a different name". A consumer parametrizes a test over
`case_index` to get one call per `--sim-runs` case, and builds its own flight-software module
against the `plant` fixture's `readers()`/`initial_state()`/`mu`. Options are namespaced
`--sim-*` so an autoloaded plugin cannot collide with a consumer's own option of the same short
name."""

import json
import secrets
import tempfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, distribution
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from Basilisk.utilities import SimulationBaseClass

from sim.basilisk.models.imu_spacecraft import ImuSpacecraftModel
from sim.basilisk.runs import write_run

# The spacecraft models this plugin knows how to build, by `--sim-spacecraft` name.
_SPACECRAFT_MODELS: dict[str, type] = {"imu_spacecraft": ImuSpacecraftModel}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--sim-runs", action="store", type=int, default=8)
    parser.addoption("--sim-out", action="store", type=str, default=None)
    parser.addoption("--sim-seed", action="store", type=int, default=0)
    parser.addoption("--sim-duration-s", action="store", type=float, default=5700.0)
    parser.addoption("--sim-spacecraft", action="store", type=str, default="imu_spacecraft")


def pytest_configure(config: pytest.Config) -> None:
    """Resolves this session's `out_dir`, but never creates it: loaded via this package's own
    `pytest11` entry point, this plugin configures every pytest session of a consumer that has
    `wise-dynamics` installed, most of which run no case and need no directory at all. The
    directory is created on first use, by `run_writer`'s own `write_run` call."""
    out = config.getoption("--sim-out")
    if out:
        out_dir = Path(out)
    else:
        workerinput = getattr(config, "workerinput", None)
        shared = workerinput.get("wise_dynamics_out_dir") if workerinput else None
        out_dir = (
            Path(shared)
            if shared
            else Path(tempfile.gettempdir()) / f"wise_dynamics_{secrets.token_hex(8)}"
        )
    config._wise_dynamics_out_dir = out_dir


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """pytest-xdist's own controller-side hook, called once per worker before it starts: hands
    the worker the controller's already-resolved `out_dir`, so every worker in a campaign writes
    to the one directory the controller picked -- with or without `--sim-out` -- rather than each
    worker falling back to its own `mkdtemp`. `optionalhook=True`: xdist defines this hook's
    spec, and loads this plugin into every session of a consumer that has `wise-dynamics`
    installed, with or without xdist present. Without it, pytest's plugin validation would
    reject this hookimpl as unknown wherever xdist's hookspec is not registered."""
    node.workerinput["wise_dynamics_out_dir"] = str(node.config._wise_dynamics_out_dir)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_index" in metafunc.fixturenames:
        runs = metafunc.config.getoption("--sim-runs")
        metafunc.parametrize("case_index", range(runs))


@pytest.fixture
def out_dir(request: pytest.FixtureRequest) -> Path:
    return request.config._wise_dynamics_out_dir


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--sim-seed")


@pytest.fixture
def campaign_size(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--sim-runs")


@pytest.fixture
def duration_s(request: pytest.FixtureRequest) -> float:
    return request.config.getoption("--sim-duration-s")


@pytest.fixture
def spacecraft_name(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--sim-spacecraft")


@pytest.fixture
def case_seeds(
    case_index: int, seed: int, campaign_size: int
) -> tuple[np.random.SeedSequence, np.random.SeedSequence]:
    """This case's own two independent child seed sequences, spawned from the campaign seed so
    no case's draws or sensor noise overlaps another's anywhere in the campaign: one for this
    case's dispersed draws, one for its sensor noise. `numpy.random.SeedSequence(seed).spawn(n)`
    is a pure function of `seed` and `n`, so every fixture in the campaign that calls it with
    this same `campaign_size` derives the same per-case children regardless of call order."""
    case_seed = np.random.SeedSequence(seed).spawn(campaign_size)[case_index]
    draws_seed, noise_seed = case_seed.spawn(2)
    return draws_seed, noise_seed


@pytest.fixture
def rng(case_seeds: tuple[np.random.SeedSequence, np.random.SeedSequence]) -> np.random.Generator:
    draws_seed, _ = case_seeds
    return np.random.default_rng(draws_seed)


@pytest.fixture
def draws(rng: np.random.Generator, spacecraft_name: str) -> dict[str, np.ndarray]:
    """The named spacecraft model's own standard per-case dispersion, as that model's
    constructor keyword arguments."""
    model_cls = _SPACECRAFT_MODELS[spacecraft_name]
    return model_cls.default_draws(rng)


@pytest.fixture
def plant(
    draws: dict[str, np.ndarray],
    spacecraft_name: str,
    case_seeds: tuple[np.random.SeedSequence, np.random.SeedSequence],
) -> Any:
    """The named spacecraft model, built with this case's dispersed draws and its own
    independent sensor-noise seed."""
    model_cls = _SPACECRAFT_MODELS[spacecraft_name]
    sim = SimulationBaseClass.SimBaseClass()
    _, noise_seed = case_seeds
    rng_seed = int(noise_seed.generate_state(1)[0])
    return model_cls(sim, rng_seed=rng_seed, **draws)


def _wise_dynamics_version() -> str:
    """This package's own installed version, read from its distribution metadata: a tagged-
    release consumer always gets this package's own version this way. Appends the pinned commit
    from the install's `direct_url.json` when one is recorded (a VCS install); a local or
    ordinary wheel install has none, and the plain version is returned."""
    try:
        dist = distribution("wise-dynamics")
    except PackageNotFoundError:
        return "unknown"

    text = dist.version
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except ValueError:
            direct_url = {}
        commit_id = direct_url.get("vcs_info", {}).get("commit_id")
        if commit_id:
            text = f"{text}+{commit_id[:12]}"
    return text


@pytest.fixture
def run_writer(case_index: int, seed: int, draws: dict[str, np.ndarray], out_dir: Path):
    """A callable writing `<out_dir>/run_<NNNN>.h5`: fills in the generic root attrs (`case`,
    the campaign `seed`, `producer`, `created_utc`, `versions_bsk`, `versions_wise_dynamics`),
    records this case's own `draws` as `draw_<name>` attrs, and merges in whatever `extra_attrs`
    the caller supplies (e.g. its own `versions_<package>` entries) before writing `arrays` via
    `write_run`. Returns the path written."""

    def write(
        arrays: dict[str, tuple[np.ndarray, str]], extra_attrs: dict[str, Any] | None = None
    ) -> Path:
        attrs: dict[str, Any] = {
            "case": case_index,
            "seed": seed,
            "producer": "wise-dynamics sim/basilisk",
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "versions_bsk": pkg_version("bsk"),
            "versions_wise_dynamics": _wise_dynamics_version(),
        }
        for name, value in draws.items():
            attrs[f"draw_{name}"] = np.asarray(value, dtype=float)
        if extra_attrs:
            attrs.update(extra_attrs)

        path = out_dir / f"run_{case_index:04d}.h5"
        write_run(path, arrays, attrs)
        return path

    return write
