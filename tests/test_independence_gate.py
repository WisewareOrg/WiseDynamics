"""`scripts/check-sim-independent.sh` against a copy of the tree, clean and seeded."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
FLIGHT_PACKAGE = "gnc" + "_algorithms"
FLIGHT_DISTRIBUTION = "wise-" + "gnc-design"


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in ("wise_dynamics", "tests", "scripts"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(ROOT / "pyproject.toml", tmp_path)
    return tmp_path


def gate(tree: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/check-sim-independent.sh"], cwd=tree, capture_output=True, text=True
    )


def test_the_clean_tree_passes(tree):
    assert gate(tree).returncode == 0


@pytest.mark.parametrize("directory", ["wise_dynamics", "tests"])
def test_a_flight_software_import_fails(tree, directory):
    (tree / directory / "leak.py").write_text(f"from {FLIGHT_PACKAGE} import mrp\n")
    result = gate(tree)
    assert result.returncode == 1
    assert FLIGHT_PACKAGE in result.stdout


def test_a_flight_software_dependency_fails(tree):
    pyproject = tree / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            '"numpy>=2.0",', f'"numpy>=2.0",\n    "{FLIGHT_DISTRIBUTION}",'
        )
    )
    assert FLIGHT_DISTRIBUTION in pyproject.read_text()
    result = gate(tree)
    assert result.returncode == 1
    assert FLIGHT_DISTRIBUTION in result.stdout
