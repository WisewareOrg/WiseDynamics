"""The `%%monte_carlo` magic and `load_runs`.

The magic's behaviours run in-process through an IPython shell against a temporary jupytext project.
One test runs the magic in a real Jupyter kernel, and one runs a real `leo500` campaign through it
and reads it back from a second kernel.
"""

import copy
import json
import os
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import h5py
import nbformat
import pytest
from IPython.core.interactiveshell import InteractiveShell
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from wise_dynamics import notebook as wd_notebook

LOAD = "%load_ext wise_dynamics.notebook"
NO_SESSION = "no Monte Carlo session is active: run pytest --sim-session <name>"
FORMATS = "ipynb,py:percent"
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


def _activate(root: Path, name: str | None) -> None:
    """Write the session pointer `<root>/monte_carlo/SESSION`, or remove it.

    Args:
        root: The pytest rootdir, which holds the `monte_carlo` root.
        name: The active session's name, or `None` for no active session.
    """
    pointer = root / "monte_carlo" / "SESSION"
    if name is None:
        pointer.unlink(missing_ok=True)
        return
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(f"{name}\n")


def _write_session(
    notebook: Path,
    name: str,
    cases: list[int],
    planned: int,
    crashed: tuple[int, ...] = (),
    stale: tuple[int, ...] = (),
) -> None:
    """Build session `name`'s folder for `TestA::test_x` of the notebook, by hand.

    Args:
        notebook: The paired notebook `sim_notebook` made.
        name: The session's name.
        cases: The cases that get a run file.
        planned: The case count `planned.json` records.
        crashed: The cases that get a crash record.
        stale: Further cases, at or above `planned`, that get a run file.
    """
    directory = notebook.parent / "monte_carlo" / name / "campaign" / "TestA" / "test_x"
    directory.mkdir(parents=True)
    (directory / "planned.json").write_text(json.dumps({"runs": planned}))
    for case in [*cases, *stale]:
        with h5py.File(directory / f"run_{case:04d}.h5", "w") as run_file:
            run_file.attrs["case"] = case
            run_file.create_dataset("t_s", data=[0.0]).attrs["units"] = "s"
    for case in crashed:
        record = {"exception": "boom", "traceback": "Traceback..."}
        (directory / f"run_{case:04d}.crash.json").write_text(json.dumps(record))


def _execute(
    directory: Path, cells: list[nbformat.NotebookNode], env: dict[str, str]
) -> nbformat.NotebookNode:
    """Run `cells` in order in a new kernel in `directory`, started with exactly `env`.

    Args:
        directory: The kernel's working directory.
        cells: The cells to run, in memory, whatever the saved notebook holds.
        env: The kernel's whole environment.

    Returns:
        The executed notebook, with each cell's outputs and the widget state. A cell that raises
        ends the run with `CellExecutionError`.
    """
    notebook = nbformat.v4.new_notebook(cells=copy.deepcopy(cells))
    client = NotebookClient(
        notebook,
        kernel_name="python3",
        timeout=900,
        resources={"metadata": {"path": str(directory)}},
    )
    client.execute(env=env)
    return notebook


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
    notebook = _execute(directory, cells, env)
    return [
        "".join(out.text for out in cell.outputs if out.output_type == "stream")
        for cell in notebook.cells
    ]


@pytest.fixture
def shell() -> Iterator[InteractiveShell]:
    """Provide a fresh in-process IPython shell with the magic loaded.

    Yields:
        InteractiveShell: The shell; the singleton is cleared before and after, so no namespace
        leaks between tests.
    """
    InteractiveShell.clear_instance()
    instance = InteractiveShell.instance()
    instance.run_line_magic("load_ext", "wise_dynamics.notebook")
    yield instance
    InteractiveShell.clear_instance()


def _magic_env(monkeypatch: pytest.MonkeyPatch, session: Path | None = None, ci: bool = False):
    """Set exactly the two variables the magic reads.

    Args:
        monkeypatch: pytest's fixture for setting the environment for this test only.
        session: What `JPY_SESSION_NAME` names, or `None` to leave it unset.
        ci: Whether `CI` is set.
    """
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("JPY_SESSION_NAME", raising=False)
    if ci:
        monkeypatch.setenv("CI", "true")
    if session:
        monkeypatch.setenv("JPY_SESSION_NAME", str(session))


def _body(cell: nbformat.NotebookNode) -> str:
    """Return a `%%monte_carlo` cell's source after the magic line.

    Args:
        cell: A cell built by `_magic`.

    Returns:
        What IPython passes the magic as `cell`.
    """
    return cell.source.partition("\n")[2]


def test_an_unsaved_cell_is_refused_and_nothing_runs(
    shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A cell that differs from the saved notebook raises the unsaved message before running.

    Args:
        shell: The in-process shell with the magic loaded.
        monkeypatch: pytest's fixture for setting the environment for this test only.
        tmp_path: The test's own temporary directory.
    """
    saved = _magic(_marker_test("test_saved", tmp_path / "saved"))
    path = _write_notebook(tmp_path, [_code(LOAD), saved])
    _magic_env(monkeypatch, path)
    edited = _body(saved) + f"\nPath({str(tmp_path / 'top_level')!r}).write_text('ran')\n"

    with pytest.raises(RuntimeError, match="unsaved changes"):
        shell.run_cell_magic("monte_carlo", "", edited)

    assert not (tmp_path / "top_level").exists()
    assert not (tmp_path / "saved").exists()


def test_an_unpaired_notebook_is_refused_with_how_to_pair_it(
    shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A notebook with neither metadata pairing nor `jupytext.toml` is refused; no `.py` appears.

    Args:
        shell: The in-process shell with the magic loaded.
        monkeypatch: pytest's fixture for setting the environment for this test only.
        tmp_path: The test's own temporary directory.
    """
    cell = _magic(_marker_test("test_here", tmp_path / "ran"))
    path = _write_notebook(tmp_path, [_code(LOAD), cell], paired=False)
    _magic_env(monkeypatch, path)

    with pytest.raises(RuntimeError, match="jupytext --set-formats ipynb,py:percent"):
        shell.run_cell_magic("monte_carlo", "", _body(cell))

    assert not path.with_suffix(".py").exists()
    assert not (tmp_path / "ran").exists()


def test_a_notebook_paired_only_by_jupytext_toml_runs(
    shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """With the pairing in `jupytext.toml` and none in the notebook, the cell's test runs.

    The kernel also sets VS Code's `__vsc_ipynb_file__` beside a `JPY_SESSION_NAME` that names no
    file, as VS Code does; the notebook is the one `__vsc_ipynb_file__` names.

    Args:
        shell: The in-process shell with the magic loaded.
        monkeypatch: pytest's fixture for setting the environment for this test only.
        tmp_path: The test's own temporary directory.
    """
    cell = _magic(_marker_test("test_here", tmp_path / "ran"))
    path = _write_notebook(tmp_path, [_code(LOAD), cell], paired=False)
    (tmp_path / "jupytext.toml").write_text(f'formats = "{FORMATS}"\n')
    _magic_env(monkeypatch, tmp_path / "campaign-a26c00bd-330f-48bb-9948-601a740c347e.ipynb")
    shell.user_ns["__vsc_ipynb_file__"] = str(path)

    shell.run_cell_magic("monte_carlo", "", _body(cell))

    assert (tmp_path / "ran").read_text() == "ran"


def test_a_notebook_given_as_its_paired_py_runs(
    shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A notebook opened as the Jupytext `.py` runs that `.py`'s test and is not re-synced.

    Args:
        shell: The in-process shell with the magic loaded.
        monkeypatch: pytest's fixture for setting the environment for this test only.
        tmp_path: The test's own temporary directory.
    """
    cell = _magic(_marker_test("test_here", tmp_path / "ran"))
    py = _sync(_write_notebook(tmp_path, [_code(LOAD), cell]))
    py.with_suffix(".ipynb").unlink()
    mtime = py.stat().st_mtime_ns
    _magic_env(monkeypatch, py)

    shell.run_cell_magic("monte_carlo", "", _body(cell))

    assert (tmp_path / "ran").read_text() == "ran"
    assert not py.with_suffix(".ipynb").exists()
    assert py.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("run", ["first", "last"])
def test_cells_after_the_last_campaign_are_marked_notebook_only_before_the_run(
    run: str, shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Unmarked code cells after the last `%%monte_carlo` cell are marked and skipped on import.

    Whichever campaign runs, a cell between the two, and a later cell already carrying `active`,
    keep their metadata; the run's workers import the re-synced `.py` without the marked cells.

    Args:
        run: Which of the two campaigns the magic runs.
        shell: The in-process shell with the magic loaded.
        monkeypatch: pytest's fixture for setting the environment for this test only.
        tmp_path: The test's own temporary directory.
    """
    first = _magic(_marker_test("test_first", tmp_path / "first"))
    last = _magic(_marker_test("test_last", tmp_path / "last"))
    imported = f"Path({str(tmp_path / 'imported')!r}).write_text('ran')"
    cells = [
        _code(LOAD),
        first,
        _code("between = 1"),
        last,
        _code(f"from pathlib import Path\n{imported}"),
        nbformat.v4.new_markdown_cell("Plots"),
        _code("kept = 1", active="ipynb,py"),
        _code("analysis = 1"),
    ]
    path = _write_notebook(tmp_path, cells)
    _magic_env(monkeypatch, path)

    shell.run_cell_magic("monte_carlo", "", _body({"first": first, "last": last}[run]))

    actives = [each.metadata.get("active") for each in nbformat.read(path, as_version=4).cells]
    assert actives == [None, None, None, None, "ipynb", None, "ipynb,py", "ipynb"]
    assert (tmp_path / run).read_text() == "ran"
    assert not (tmp_path / "imported").exists()


def test_a_notebook_with_nothing_to_mark_is_not_rewritten(
    shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """With every cell after the campaign already marked, the saved `.ipynb` is left untouched.

    Args:
        shell: The in-process shell with the magic loaded.
        monkeypatch: pytest's fixture for setting the environment for this test only.
        tmp_path: The test's own temporary directory.
    """
    cell = _magic(_marker_test("test_here", tmp_path / "ran"))
    path = _write_notebook(tmp_path, [_code(LOAD), cell, _code("analysis = 1", active="ipynb")])
    _sync(path)
    _sync(path)  # jupytext's second sync rewrites both files; the third leaves them alone
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    _magic_env(monkeypatch, path)

    shell.run_cell_magic("monte_carlo", "", _body(cell))

    assert (tmp_path / "ran").read_text() == "ran"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_a_cell_defining_no_test_is_refused(
    shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A saved cell with no top-level test raises, and its code does not run.

    Args:
        shell: The in-process shell with the magic loaded.
        monkeypatch: pytest's fixture for setting the environment for this test only.
        tmp_path: The test's own temporary directory.
    """
    cell = _magic(f"from pathlib import Path\nPath({str(tmp_path / 'ran')!r}).write_text('ran')")
    path = _write_notebook(tmp_path, [_code(LOAD), cell])
    _magic_env(monkeypatch, path)

    with pytest.raises(RuntimeError, match="defines no test"):
        shell.run_cell_magic("monte_carlo", "", _body(cell))

    assert not (tmp_path / "ran").exists()


def test_no_path_source_is_refused_asking_for_the_path(
    shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch
):
    """With no argument, no `JPY_SESSION_NAME` and no `__vsc_ipynb_file__`, the magic asks for one.

    Args:
        shell: The in-process shell with the magic loaded.
        monkeypatch: pytest's fixture for setting the environment for this test only.
    """
    _magic_env(monkeypatch)

    with pytest.raises(RuntimeError, match="pass it as an argument"):
        shell.run_cell_magic("monte_carlo", "", "def test_x():\n    pass\n")


@pytest.mark.parametrize("ci", [True, False], ids=["ci", "no-ci"])
def test_the_run_gets_the_ci_option_only_when_ci_is_set(
    shell: InteractiveShell, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, ci: bool
):
    """The cell's test sees `--ci` exactly when `CI` is set.

    Args:
        shell: The in-process shell with the magic loaded.
        monkeypatch: pytest's fixture for setting the environment for this test only.
        tmp_path: The test's own temporary directory.
        ci: Whether `CI` is set.
    """
    sentinel = tmp_path / "ci"
    cell = _magic(
        "from pathlib import Path\n\n\n"
        "def test_records_ci(request):\n"
        f"    Path({str(sentinel)!r}).write_text(repr(request.config.getoption('--ci')))\n"
    )
    path = _write_notebook(tmp_path, [_code(LOAD), cell])
    _magic_env(monkeypatch, path, ci=ci)

    shell.run_cell_magic("monte_carlo", "", _body(cell))

    assert sentinel.read_text() == repr(ci)


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
        directory: The test's directory within its session.

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

    The run goes into the active session `real`. It prints no assert-rewrite warning, and its
    progress is one display, updated in place until it reads four of four.

    Args:
        tmp_path: The test's own temporary directory.
    """
    campaign = [_code(LOAD), nbformat.v4.new_code_cell(CAMPAIGN.strip("\n"))]
    path = _write_notebook(tmp_path, campaign)
    _activate(tmp_path, "real")
    env = _env(path)
    notebook = _execute(tmp_path, campaign, env)
    streams = "".join(out.text for out in notebook.cells[1].outputs if out.output_type == "stream")
    meter = [
        out.data["text/plain"]
        for out in notebook.cells[1].outputs
        if out.output_type == "display_data"
    ]

    output = _run(tmp_path, [_code(LOAD), _code(SUMMARY, active="ipynb")], env)[1]
    (line,) = [line for line in output.splitlines() if line.startswith("SUMMARY ")]
    loaded = json.loads(line.removeprefix("SUMMARY "))

    expected = _expected_runs(tmp_path / "monte_carlo" / "real" / "campaign" / "test_leo")
    assert "PytestAssertRewriteWarning" not in streams
    assert len(meter) == 1
    assert "4/4" in meter[-1]
    assert re.search(r"^real: 4 of 4 cases$", output, flags=re.MULTILINE)
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
def sim_notebook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Save and sync a paired notebook defining `TestA.test_x`, `TestA.test_y` and `test_x`.

    Args:
        tmp_path: The test's own temporary directory.
        monkeypatch: pytest's fixture; it unsets `CI` for the test.

    Returns:
        The notebook's path; nothing has run, so there are no run files. Session `s` is active.
    """
    monkeypatch.delenv("CI", raising=False)
    path = _write_notebook(tmp_path, [_code(SIM_TESTS)])
    _sync(path)
    _activate(tmp_path, "s")
    return path


def test_load_runs_with_no_run_files_names_the_expected_folder(sim_notebook: Path):
    """With no folder in the active session, `load_runs` raises `FileNotFoundError` naming it.

    Args:
        sim_notebook: The paired notebook, never run.
    """
    folder = sim_notebook.parent / "monte_carlo" / "s" / "campaign" / "TestA" / "test_x"

    with pytest.raises(FileNotFoundError, match=re.escape(str(folder))):
        wd_notebook.load_runs("TestA::test_x", sim_notebook)


def test_load_runs_reads_the_active_session_and_reports_found_planned_and_crashed(
    sim_notebook: Path, capsys: pytest.CaptureFixture[str]
):
    """Cases 0, 1 and 3 of 4 are returned, case 2's crash is reported, a stale case 9 is ignored.

    Args:
        sim_notebook: The paired notebook, never run.
        capsys: pytest's fixture for capturing what `load_runs` prints.
    """
    _write_session(sim_notebook, "s", [0, 1, 3], 4, crashed=(2,), stale=(9,))

    runs = wd_notebook.load_runs("TestA::test_x", sim_notebook)

    assert [int(run["attrs"]["case"]) for run in runs] == [0, 1, 3]
    assert capsys.readouterr().out == "s: 3 of 4 cases, crashed: 2\n"


def test_load_runs_follows_a_session_switch_within_one_process(
    sim_notebook: Path, capsys: pytest.CaptureFixture[str]
):
    """Pointing at another session between two calls makes the second call read that session.

    Args:
        sim_notebook: The paired notebook, never run.
        capsys: pytest's fixture for capturing what `load_runs` prints.
    """
    _write_session(sim_notebook, "first", [0], 1)
    _write_session(sim_notebook, "second", [0, 1], 2)
    _activate(sim_notebook.parent, "first")

    assert len(wd_notebook.load_runs("TestA::test_x", sim_notebook)) == 1
    assert capsys.readouterr().out == "first: 1 of 1 cases\n"

    _activate(sim_notebook.parent, "second")

    assert len(wd_notebook.load_runs("TestA::test_x", sim_notebook)) == 2
    assert capsys.readouterr().out == "second: 2 of 2 cases\n"


def test_load_runs_with_no_active_session_names_how_to_pick_one(sim_notebook: Path):
    """With no pointer, `load_runs` raises `RuntimeError` naming `--sim-session`.

    Args:
        sim_notebook: The paired notebook, never run.
    """
    _write_session(sim_notebook, "s", [0], 1)
    _activate(sim_notebook.parent, None)

    with pytest.raises(RuntimeError, match=re.escape(NO_SESSION)):
        wd_notebook.load_runs("TestA::test_x", sim_notebook)


def test_load_runs_with_an_invalid_name_in_the_pointer_raises_that_error(sim_notebook: Path):
    """A corrupt pointer is reported as such, not as "no session is active".

    Args:
        sim_notebook: The paired notebook, never run.
    """
    _activate(sim_notebook.parent, "..")

    with pytest.raises(RuntimeError, match="invalid Monte Carlo session name") as raised:
        wd_notebook.load_runs("TestA::test_x", sim_notebook)

    assert NO_SESSION not in str(raised.value)


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
