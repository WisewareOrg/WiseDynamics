"""The `%%monte_carlo` magic and `load_runs`.

The magic's mechanics are plain unit tests of `wise_dynamics.notebook`'s functions. One test runs
the magic in a real Jupyter kernel, and one runs a real `leo500` campaign through it and reads it
back from a second kernel.
"""

import copy
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from wise_dynamics import notebook as wd_notebook

LOAD = "%load_ext wise_dynamics.notebook"
FORMATS = "ipynb,py:percent"
PY_NEWER = (
    "The .py was edited after the notebook was saved; reload the notebook from disk before running"
)
CAMPAIGN = """
%%monte_carlo
import os
import re

import pytest
from Basilisk.fswAlgorithms import inertial3D


class Guidance(inertial3D.inertial3D):
    period_s = 1.0


@pytest.mark.sim_runs(4)
def test_leo(sim_case_index, sim_duration_s, sim_scenario, sim_run_file):
    datasets = sim_scenario.run(Guidance(), {}, duration_s=sim_duration_s)
    sim_run_file(datasets, {"worker_pid": os.getpid()})
"""
SUMMARY = """
import json
import os
import re

from wise_dynamics.notebook import load_runs

runs = load_runs("test_leo")
print("SUMMARY", json.dumps({
    "kernel_pid": os.getpid(),
    "cases": [int(run["attrs"]["case"]) for run in runs],
    "worker_pids": [int(run["attrs"]["worker_pid"]) for run in runs],
    "units": [{key: units for key, (_, units) in run["datasets"].items()} for run in runs],
    "sums": [
        {key: float(values.sum()) for key, (values, _) in run["datasets"].items()}
        for run in runs
    ],
}))
"""
SIM_TESTS = """
class TestA:
    def test_x(self, sim_case_index):
        pass

    def test_y(self, sim_case_index):
        pass


def test_x(sim_case_index):
    pass
"""


def _env(session: Path) -> dict[str, str]:
    """Build a kernel's whole environment, with `CI` unset and `JPY_SESSION_NAME` set.

    Args:
        session: The notebook `JPY_SESSION_NAME` names.

    Returns:
        The inherited environment without `CI`, `JPY_SESSION_NAME` and pytest's own variables,
        plus `JPY_SESSION_NAME`.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("CI", "JPY_SESSION_NAME") and not key.startswith("PYTEST_")
    }
    env["JPY_SESSION_NAME"] = str(session)
    return env


def _code(source: str, **metadata: str) -> nbformat.NotebookNode:
    """Build a code cell.

    Args:
        source: The cell's source, dedented, without its leading newline.
        **metadata: The cell's metadata, such as `active="ipynb"`.

    Returns:
        The cell.
    """
    return nbformat.v4.new_code_cell(textwrap.dedent(source).strip("\n"), metadata=metadata)


def _magic(body: str) -> nbformat.NotebookNode:
    """Build a `%%monte_carlo` cell.

    Args:
        body: The cell's source after the magic line.

    Returns:
        The cell.
    """
    return nbformat.v4.new_code_cell("%%monte_carlo\n" + textwrap.dedent(body).strip("\n"))


def _marker_test(name: str, marker: Path) -> str:
    """Write the source of a plain test that writes `marker` when it runs.

    Args:
        name: The test function's name.
        marker: The file the test writes.

    Returns:
        The test's source, with `Path` imported.
    """
    write = f"Path({str(marker)!r}).write_text('ran')"
    return f"from pathlib import Path\n\n\ndef {name}():\n    {write}\n"


def _write_notebook(
    directory: Path, cells: list[nbformat.NotebookNode], paired: bool = True
) -> Path:
    """Save a notebook, with a `pyproject.toml` that makes `directory` the pytest rootdir.

    Args:
        directory: Where the notebook, `campaign.ipynb`, is written.
        cells: The notebook's cells.
        paired: Whether the notebook is paired with `campaign.py` as jupytext's percent format.

    Returns:
        The notebook's path.
    """
    (directory / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    if paired:
        notebook.metadata["jupytext"] = {"formats": FORMATS}
    path = directory / "campaign.ipynb"
    nbformat.write(notebook, path)
    return path


def _sync(notebook: Path) -> Path:
    """Write the paired `.py` from the saved notebook, as jupytext's own CLI does.

    Args:
        notebook: The saved, paired notebook.

    Returns:
        The `.py` it writes.
    """
    subprocess.run(
        [sys.executable, "-m", "jupytext", "--sync", str(notebook)],
        check=True,
        capture_output=True,
    )
    return notebook.with_suffix(".py")


def _later(path: Path, seconds: int = 60) -> None:
    """Make `path` newer than anything written so far.

    Args:
        path: The file whose modification time is set.
        seconds: How far past now to set it.
    """
    stamp = path.stat().st_mtime + seconds
    os.utime(path, (stamp, stamp))


def _run(directory: Path, cells: list[nbformat.NotebookNode], env: dict[str, str]) -> list[str]:
    """Run `cells` in order in a new kernel in `directory`, started with exactly `env`.

    Args:
        directory: The kernel's working directory.
        cells: The cells to run, in memory, whatever the saved notebook holds.
        env: The kernel's whole environment.

    Returns:
        What each cell wrote to standard output. A cell that raises ends the run with
        `CellExecutionError`.
    """
    notebook = nbformat.v4.new_notebook(cells=copy.deepcopy(cells))
    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=900,
        resources={"metadata": {"path": str(directory)}},
    )
    client.execute(env=env)
    return [
        "".join(out.text for out in cell.outputs if out.output_type == "stream")
        for cell in notebook.cells
    ]


def test_a_cells_tests_are_its_top_level_test_functions_and_classes():
    """Only top-level `test*` functions and `Test*` classes count, async ones too."""
    source = textwrap.dedent(
        """
        def test_a():
            def test_nested():
                pass

        async def test_b():
            pass

        class TestC:
            def test_method(self):
                pass

        class Helper:
            pass

        def helper():
            pass
        """
    )
    assert wd_notebook._test_names(source) == ["test_a", "test_b", "TestC"]


def test_a_cell_with_no_test_names_none():
    """A cell defining no test yields no names, which is what the magic refuses."""
    assert wd_notebook._test_names("value = 1\n\ndef helper():\n    pass\n") == []


def test_a_cell_differing_from_the_saved_one_only_in_whitespace_counts_as_saved(tmp_path: Path):
    """An edited cell is not among the saved bodies; whitespace-only differences do not count.

    Trailing whitespace and leading and trailing blank lines are ignored.

    Args:
        tmp_path: The test's own temporary directory.
    """
    body = _marker_test("test_saved", tmp_path / "saved")
    path = _write_notebook(tmp_path, [_code(LOAD), _magic(body)])
    saved = wd_notebook._saved_magic_bodies(path)

    padded = "\n\n" + "\n".join(line + "  " for line in body.split("\n")) + "\n\n\n"
    edited = body.replace("test_saved", "test_edited")
    assert padded != body
    assert wd_notebook._normalized(body) in saved
    assert wd_notebook._normalized(padded) in saved
    assert wd_notebook._normalized(edited) not in saved


def test_a_saved_edit_not_yet_in_the_py_is_synced(tmp_path: Path):
    """A saved notebook newer than its `.py` is written into it.

    Args:
        tmp_path: The test's own temporary directory.
    """
    path = _write_notebook(tmp_path, [_code(LOAD), _magic(_marker_test("test_first", tmp_path))])
    py = _sync(path)
    _write_notebook(tmp_path, [_code(LOAD), _magic(_marker_test("test_second", tmp_path))])
    _later(path)

    wd_notebook._sync_py(path, py)

    assert "test_second" in py.read_text()
    assert "test_first" not in py.read_text()


def test_a_py_newer_than_the_notebook_with_other_content_is_refused(tmp_path: Path):
    """A `.py` that differs from the notebook and was edited after it was saved is not overwritten.

    Args:
        tmp_path: The test's own temporary directory.
    """
    path = _write_notebook(tmp_path, [_code(LOAD), _magic(_marker_test("test_here", tmp_path))])
    py = _sync(path)
    py.write_text(py.read_text() + "\n# edited in the py\n")
    _later(py)

    with pytest.raises(RuntimeError, match=PY_NEWER):
        wd_notebook._sync_py(path, py)

    assert "# edited in the py" in py.read_text()


def test_a_py_newer_than_the_notebook_with_the_same_content_is_in_sync(tmp_path: Path):
    """A `.py` that is newer but renders the same as the notebook is accepted untouched.

    Args:
        tmp_path: The test's own temporary directory.
    """
    path = _write_notebook(tmp_path, [_code(LOAD), _magic(_marker_test("test_here", tmp_path))])
    py = _sync(path)
    _later(py)
    before = py.read_text()

    wd_notebook._sync_py(path, py)

    assert py.read_text() == before


def test_an_unpaired_notebook_is_refused_with_how_to_pair_it(tmp_path: Path):
    """A notebook with no paired `.py` raises a message naming jupytext's pairing command.

    Args:
        tmp_path: The test's own temporary directory.
    """
    path = _write_notebook(tmp_path, [_code(LOAD)], paired=False)

    with pytest.raises(RuntimeError, match="jupytext --set-formats ipynb,py:percent"):
        wd_notebook._paired_py(path)


@pytest.mark.parametrize(
    ("line", "session", "vscode", "expected"),
    [
        (" arg.ipynb ", "session.ipynb", "vscode.ipynb", "arg.ipynb"),
        ("", "session.ipynb", "vscode.ipynb", "session.ipynb"),
        ("", None, "vscode.ipynb", "vscode.ipynb"),
        ("", None, None, None),
    ],
    ids=["argument", "session-name", "vscode", "missing"],
)
def test_the_notebook_path_comes_from_the_argument_then_the_session_then_vscode(
    monkeypatch: pytest.MonkeyPatch,
    line: str,
    session: str | None,
    vscode: str | None,
    expected: str | None,
):
    """The path is the line argument, else `JPY_SESSION_NAME`, else `__vsc_ipynb_file__`.

    With none of them, the magic asks for a path.

    Args:
        monkeypatch: pytest's fixture for setting the environment for this test only.
        line: The magic's line argument.
        session: The value of `JPY_SESSION_NAME`, or `None` to leave it unset.
        vscode: The value of `__vsc_ipynb_file__` in the namespace, or `None` to leave it unset.
        expected: The path found, or `None` when the call must raise.
    """
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)
    if session:
        monkeypatch.setenv("JPY_SESSION_NAME", session)
    user_ns = {"__vsc_ipynb_file__": vscode} if vscode else {}

    if expected is None:
        with pytest.raises(RuntimeError, match="path"):
            wd_notebook._notebook_path(line, user_ns)
    else:
        assert wd_notebook._notebook_path(line, user_ns) == Path(expected)


@pytest.mark.parametrize("ci", [True, False], ids=["ci", "no-ci"])
def test_the_ci_flag_is_passed_only_when_ci_is_set(monkeypatch: pytest.MonkeyPatch, ci: bool):
    """The pytest arguments carry `--ci` exactly when `CI` is set, and run in xdist workers.

    Args:
        monkeypatch: pytest's fixture for setting the environment for this test only.
        ci: Whether `CI` is set.
    """
    monkeypatch.delenv("CI", raising=False)
    if ci:
        monkeypatch.setenv("CI", "true")

    args = wd_notebook._pytest_args(Path("campaign.py"), ["test_a", "TestB"])

    assert args[:2] == ["campaign.py::test_a", "campaign.py::TestB"]
    assert args[2:4] == ["-n", "auto"]
    assert ("--ci" in args) is ci


def test_the_magic_runs_only_its_cells_tests_in_workers_and_raises_on_failure(tmp_path: Path):
    """In a real kernel the magic runs its own cell's tests only, and raises on a failing one.

    The workers import the `.py`, which holds neither the magic nor an `active="ipynb"` cell: the
    sentinel that cell writes holds one pid, the kernel's. The cell after the failing one never
    runs.

    Args:
        tmp_path: The test's own temporary directory.
    """
    marks = tmp_path / "marks"
    marks.mkdir()
    sentinel = tmp_path / "sentinel"
    body = (
        _marker_test("test_this_cell", marks / "this") + "\n\ndef test_fails():\n    assert False\n"
    )
    cells = [
        _code(LOAD),
        _code(
            f"""
            import os

            with open({str(sentinel)!r}, "a") as file:
                file.write(f"{{os.getpid()}}\\n")
            """,
            active="ipynb",
        ),
        _code(_marker_test("test_other_cell", marks / "other")),
        _magic(body),
        _code(
            f"from pathlib import Path\nPath({str(marks / 'after')!r}).write_text('ran')",
            active="ipynb",
        ),
    ]
    path = _write_notebook(tmp_path, cells)

    with pytest.raises(CellExecutionError):
        _run(tmp_path, cells, _env(path))

    assert sorted(item.name for item in marks.iterdir()) == ["this"]
    assert len(sentinel.read_text().split()) == 1


def _expected_runs(directory: Path) -> list[dict]:
    """Read each run file in `directory` with h5py, in case order.

    Args:
        directory: The test's run directory.

    Returns:
        One mapping per file: its `case`, and the `units` and `sums` of every dataset by path.
    """
    runs = []
    for path in sorted(directory.glob("run_*.h5")):
        datasets: dict[str, h5py.Dataset] = {}
        with h5py.File(path, "r") as run_file:
            run_file.visititems(
                lambda name, node, found=datasets: (
                    found.update({name: node}) if isinstance(node, h5py.Dataset) else None
                )
            )
            runs.append(
                {
                    "case": int(run_file.attrs["case"]),
                    "units": {name: node.attrs["units"] for name, node in datasets.items()},
                    "sums": {name: float(node[:].sum()) for name, node in datasets.items()},
                }
            )
    return runs


def test_a_real_campaign_runs_in_workers_and_a_second_kernel_loads_it(tmp_path: Path):
    """A `leo500` campaign of four cases runs across workers; a fresh kernel loads every run.

    Args:
        tmp_path: The test's own temporary directory.
    """
    campaign = [_code(LOAD), nbformat.v4.new_code_cell(CAMPAIGN.strip("\n"))]
    path = _write_notebook(tmp_path, campaign)
    env = _env(path)
    _run(tmp_path, campaign, env)

    (line,) = [
        line
        for line in _run(tmp_path, [_code(LOAD), _code(SUMMARY, active="ipynb")], env)[
            1
        ].splitlines()
        if line.startswith("SUMMARY ")
    ]
    loaded = json.loads(line.removeprefix("SUMMARY "))

    expected = _expected_runs(tmp_path / "monte_carlo" / "campaign" / "test_leo")
    assert loaded["cases"] == [0, 1, 2, 3]
    assert len(set(loaded["worker_pids"])) > 1
    assert loaded["kernel_pid"] not in loaded["worker_pids"]
    assert loaded["units"] == [run["units"] for run in expected]
    assert loaded["sums"] == [run["sums"] for run in expected]


class _Holder:
    """A class pytest does not collect, whose method `load_runs` takes as a test object."""

    def test_x(self) -> None:
        """Stand in for a method defined in a notebook; it is never meant to run."""


@pytest.fixture
def sim_notebook(tmp_path: Path) -> Path:
    """Save and sync a paired notebook defining `TestA.test_x`, `TestA.test_y` and `test_x`.

    Args:
        tmp_path: The test's own temporary directory.

    Returns:
        The notebook's path; nothing has run, so there are no run files.
    """
    path = _write_notebook(tmp_path, [_code(SIM_TESTS)])
    _sync(path)
    return path


def test_load_runs_with_no_run_files_names_the_expected_folder(sim_notebook: Path):
    """With no run files, `load_runs` raises `FileNotFoundError` naming the method's folder.

    Args:
        sim_notebook: The paired notebook, never run.
    """
    folder = sim_notebook.parent / "monte_carlo" / "campaign" / "TestA" / "test_x"

    with pytest.raises(FileNotFoundError, match=re.escape(str(folder))):
        wd_notebook.load_runs("TestA::test_x", sim_notebook)


def test_load_runs_turns_a_methods_qualified_name_into_its_node_path(sim_notebook: Path):
    """A method object is looked up as `Class::method`, so a class the `.py` lacks is named so.

    Args:
        sim_notebook: The paired notebook, never run.
    """
    with pytest.raises(RuntimeError, match="_Holder::test_x does not collect"):
        wd_notebook.load_runs(_Holder.test_x, sim_notebook)


def test_load_runs_names_a_test_that_does_not_collect(sim_notebook: Path):
    """A node that is not in the `.py` raises an error naming it, not a missing-folder error.

    Args:
        sim_notebook: The paired notebook, never run.
    """
    with pytest.raises(RuntimeError, match="test_absent does not collect"):
        wd_notebook.load_runs("test_absent", sim_notebook)


def test_load_runs_refuses_a_node_with_more_than_one_run_directory(sim_notebook: Path):
    """A class collects two methods, each in its own run directory, so `load_runs` refuses it.

    Args:
        sim_notebook: The paired notebook, never run.
    """
    with pytest.raises(RuntimeError, match="2 run directories"):
        wd_notebook.load_runs("TestA", sim_notebook)
