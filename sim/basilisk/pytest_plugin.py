"""A pytest plugin exposing Monte Carlo campaign options and fixtures over this repository's
plant models, with no knowledge of any flight software. A consumer loads it with
`pytest_plugins = ["sim.basilisk.pytest_plugin"]` in its own `conftest.py`, parametrizes a test
over `case_index` to get one call per `--runs` case, and builds its own flight-software module
against the `plant` fixture's `readers()`/`initial_state()`/`mu`."""

import subprocess
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from Basilisk.utilities import SimulationBaseClass

from sim.basilisk.models.imu_spacecraft import ImuSpacecraftModel
from sim.basilisk.runs import write_run

# The spacecraft models this plugin knows how to build, by `--spacecraft` name.
_SPACECRAFT_MODELS: dict[str, type] = {"imu_spacecraft": ImuSpacecraftModel}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runs", action="store", type=int, default=8)
    parser.addoption("--out", action="store", type=str, default=None)
    parser.addoption("--seed", action="store", type=int, default=0)
    parser.addoption("--duration-s", action="store", type=float, default=5700.0)
    parser.addoption("--spacecraft", action="store", type=str, default="imu_spacecraft")


def pytest_configure(config: pytest.Config) -> None:
    out = config.getoption("--out")
    out_dir = Path(out) if out else Path(tempfile.mkdtemp(prefix="wise_dynamics_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    config._wise_dynamics_out_dir = out_dir


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case_index" in metafunc.fixturenames:
        runs = metafunc.config.getoption("--runs")
        metafunc.parametrize("case_index", range(runs))


@pytest.fixture
def out_dir(request: pytest.FixtureRequest) -> Path:
    return request.config._wise_dynamics_out_dir


@pytest.fixture
def seed(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--seed")


@pytest.fixture
def duration_s(request: pytest.FixtureRequest) -> float:
    return request.config.getoption("--duration-s")


@pytest.fixture
def spacecraft_name(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--spacecraft")


@pytest.fixture
def case_seed(case_index: int, seed: int) -> int:
    return seed + case_index


@pytest.fixture
def rng(case_seed: int) -> np.random.Generator:
    return np.random.default_rng(case_seed)


@pytest.fixture
def draws(rng: np.random.Generator, spacecraft_name: str) -> dict[str, np.ndarray]:
    """The named spacecraft model's own standard per-case dispersion, as that model's
    constructor keyword arguments."""
    model_cls = _SPACECRAFT_MODELS[spacecraft_name]
    return model_cls.default_draws(rng)


@pytest.fixture
def plant(draws: dict[str, np.ndarray], spacecraft_name: str) -> Any:
    """The named spacecraft model, built with this case's dispersed draws."""
    model_cls = _SPACECRAFT_MODELS[spacecraft_name]
    sim = SimulationBaseClass.SimBaseClass()
    return model_cls(sim, **draws)


def _git_describe(path: Path) -> str:
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@pytest.fixture
def run_writer(case_index: int, case_seed: int, out_dir: Path):
    """A callable writing `<out_dir>/run_<NNNN>.h5`: fills in the generic root attrs (`case`,
    `seed`, `producer`, `created_utc`, `versions_bsk`, `versions_wise_dynamics`) and merges in
    whatever `extra_attrs` the caller supplies (e.g. its own `draw_<name>` and
    `versions_<package>` entries) before writing `arrays` via `write_run`. Returns the path
    written."""
    repo_root = Path(__file__).resolve().parents[1]

    def write(
        arrays: dict[str, tuple[np.ndarray, str]], extra_attrs: dict[str, Any] | None = None
    ) -> Path:
        attrs: dict[str, Any] = {
            "case": case_index,
            "seed": case_seed,
            "producer": "wise-dynamics sim/basilisk",
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "versions_bsk": pkg_version("bsk"),
            "versions_wise_dynamics": _git_describe(repo_root),
        }
        if extra_attrs:
            attrs.update(extra_attrs)

        path = out_dir / f"run_{case_index:04d}.h5"
        write_run(path, arrays, attrs)
        return path

    return write
