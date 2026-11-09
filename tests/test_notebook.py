"""The `%%monte_carlo` magic and `load_runs`, exercised through real Jupyter kernels.

Each test writes a jupytext-paired notebook with its own `pyproject.toml` into `tmp_path`, runs
its cells in a kernel started with an explicit environment, and reads what the cells printed.
The mechanics tests use plain pytest tests; the end-to-end tests run a real `leo500` campaign.
"""

import copy
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

LOAD = "%load_ext wise_dynamics.notebook"
FORMATS = "ipynb,py:percent"
UNSAVED = "This cell has unsaved changes. Save the notebook and run the cell again."
PY_NEWER = (
    "The .py was edited after the notebook was saved; reload the notebook from disk before running"
)
NO_TEST = "This cell defines no test."
CAMPAIGN = """
%%monte_carlo
import os

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

from wise_dynamics.notebook import load_runs

runs = load_runs(CALL)
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


def _env(ci: bool = False, session: Path | None = None) -> dict[str, str]:
    """Build a kernel's whole environment, with `CI` and `JPY_SESSION_NAME` set or unset as asked.

    Args:
        ci: Whether `CI` is set to `true`.
        session: The notebook `JPY_SESSION_NAME` names, or `None` to leave it unset.

    Returns:
        The inherited environment without `CI`, `JPY_SESSION_NAME` and pytest's own variables,
        plus the two as asked.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("CI", "JPY_SESSION_NAME") and not key.startswith("PYTEST_")
    }
    if ci:
        env["CI"] = "true"
    if session is not None:
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
        `CellExecutionError`, and the cells after it do not run.
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


def _raised(
    directory: Path, cells: list[nbformat.NotebookNode], env: dict[str, str]
) -> CellExecutionError:
    """Run `cells`, which must raise.

    Args:
        directory: The kernel's working directory.
        cells: The cells to run.
        env: The kernel's whole environment.

    Returns:
        The error the failing cell raised.
    """
    with pytest.raises(CellExecutionError) as error:
        _run(directory, cells, env)
    return error.value


def _magic(body: str, line: str = "") -> nbformat.NotebookNode:
    """Build a `%%monte_carlo` cell.

    Args:
        body: The cell's source after the magic line.
        line: What follows `%%monte_carlo` on its first line.

    Returns:
        The cell.
    """
    return nbformat.v4.new_code_cell(f"%%monte_carlo{line}\n" + textwrap.dedent(body).strip("\n"))


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


def test_only_the_magic_cells_tests_run(tmp_path: Path):
    """A test defined in another cell is not run; the magic cell's own is.

    Args:
        tmp_path: The test's own temporary directory.
    """
    marks = tmp_path / "marks"
    marks.mkdir()
    cells = [
        _code(LOAD),
        _code(_marker_test("test_other_cell", marks / "other")),
        _magic(_marker_test("test_this_cell", marks / "this")),
    ]
    path = _write_notebook(tmp_path, cells)
    _run(tmp_path, cells, _env(session=path))

    assert sorted(item.name for item in marks.iterdir()) == ["this"]


def test_workers_import_the_py_without_the_magic_or_an_ipynb_only_cell(tmp_path: Path):
    """The sentinel an `active="ipynb"` cell writes records the kernel's pid and no worker's.

    Args:
        tmp_path: The test's own temporary directory.
    """
    sentinel = tmp_path / "sentinel"
    cells = [
        _code(LOAD),
        _code(
            f"""
            import os

            with open({str(sentinel)!r}, "a") as file:
                file.write(f"{{os.getpid()}}\\n")
            print(os.getpid())
            """,
            active="ipynb",
        ),
        _magic(_marker_test("test_here", tmp_path / "here")),
    ]
    path = _write_notebook(tmp_path, cells)
    outputs = _run(tmp_path, cells, _env(session=path))

    assert (tmp_path / "here").read_text() == "ran"
    assert sentinel.read_text().split() == [outputs[1].strip()]


@pytest.mark.parametrize("ci", [True, False], ids=["ci", "no-ci"])
def test_the_ci_flag_is_passed_only_when_ci_is_set(tmp_path: Path, ci: bool):
    """The tests see `--ci` exactly when the kernel's environment sets `CI`.

    Args:
        tmp_path: The test's own temporary directory.
        ci: Whether the kernel's environment sets `CI`.
    """
    out = tmp_path / "ci_option"
    cells = [
        _code(LOAD),
        _magic(
            f"""
            from pathlib import Path

            def test_flag(request):
                Path({str(out)!r}).write_text(str(request.config.getoption("ci")))
            """
        ),
    ]
    path = _write_notebook(tmp_path, cells)
    _run(tmp_path, cells, _env(ci=ci, session=path))

    assert out.read_text() == str(ci)


def test_an_unsaved_cell_is_refused(tmp_path: Path):
    """A cell edited in memory since the save is refused, and none of its tests run.

    Args:
        tmp_path: The test's own temporary directory.
    """
    saved = _magic(_marker_test("test_saved", tmp_path / "saved"))
    edited = _magic(_marker_test("test_edited", tmp_path / "edited"))
    path = _write_notebook(tmp_path, [_code(LOAD), saved])
    error = _raised(tmp_path, [_code(LOAD), edited], _env(session=path))

    assert error.evalue == UNSAVED
    assert not (tmp_path / "saved").exists()
    assert not (tmp_path / "edited").exists()


def test_a_cell_differing_only_in_whitespace_counts_as_saved(tmp_path: Path):
    """Trailing whitespace and leading and trailing blank lines do not make a cell unsaved.

    Args:
        tmp_path: The test's own temporary directory.
    """
    saved = _magic(_marker_test("test_saved", tmp_path / "saved"))
    magic_line, *body = saved.source.split("\n")
    padded = nbformat.v4.new_code_cell(
        magic_line + "\n\n" + "\n".join(line + "  " for line in body) + "\n\n\n"
    )
    assert padded.source != saved.source
    path = _write_notebook(tmp_path, [_code(LOAD), saved])
    _run(tmp_path, [_code(LOAD), padded], _env(session=path))

    assert (tmp_path / "saved").read_text() == "ran"


def test_a_saved_edit_not_yet_in_the_py_is_synced_and_run(tmp_path: Path):
    """A saved notebook newer than its `.py` is synced into it, and the new test runs.

    Args:
        tmp_path: The test's own temporary directory.
    """
    first = [_code(LOAD), _magic(_marker_test("test_first", tmp_path / "first"))]
    path = _write_notebook(tmp_path, first)
    _sync(path)
    second = [_code(LOAD), _magic(_marker_test("test_second", tmp_path / "second"))]
    _write_notebook(tmp_path, second)
    _later(path)
    _run(tmp_path, second, _env(session=path))

    assert (tmp_path / "second").read_text() == "ran"
    assert not (tmp_path / "first").exists()


def test_a_py_newer_than_the_notebook_is_refused(tmp_path: Path):
    """A `.py` edited after the notebook was saved is refused, and nothing runs.

    Args:
        tmp_path: The test's own temporary directory.
    """
    cells = [_code(LOAD), _magic(_marker_test("test_here", tmp_path / "here"))]
    path = _write_notebook(tmp_path, cells)
    _later(_sync(path))
    error = _raised(tmp_path, cells, _env(session=path))

    assert PY_NEWER in error.evalue
    assert not (tmp_path / "here").exists()


def test_an_unpaired_notebook_is_refused_with_how_to_pair_it(tmp_path: Path):
    """A notebook with no paired `.py` raises a message naming jupytext's pairing command.

    Args:
        tmp_path: The test's own temporary directory.
    """
    cells = [_code(LOAD), _magic(_marker_test("test_here", tmp_path / "here"))]
    path = _write_notebook(tmp_path, cells, paired=False)
    error = _raised(tmp_path, cells, _env(session=path))

    assert "jupytext --set-formats ipynb,py:percent" in error.evalue
    assert not (tmp_path / "here").exists()


def test_a_cell_that_defines_no_test_is_refused(tmp_path: Path):
    """A magic cell with no `test*` function or `Test*` class raises its message.

    Args:
        tmp_path: The test's own temporary directory.
    """
    cells = [_code(LOAD), _magic("value = 1")]
    path = _write_notebook(tmp_path, cells)
    error = _raised(tmp_path, cells, _env(session=path))

    assert NO_TEST in error.evalue


def test_a_missing_notebook_path_is_refused(tmp_path: Path):
    """With no argument, `JPY_SESSION_NAME` or `__vsc_ipynb_file__`, the magic asks for a path.

    Args:
        tmp_path: The test's own temporary directory.
    """
    cells = [_code(LOAD), _magic(_marker_test("test_here", tmp_path / "here"))]
    _write_notebook(tmp_path, cells)
    error = _raised(tmp_path, cells, _env())

    assert "path" in error.evalue.lower()
    assert not (tmp_path / "here").exists()


@pytest.mark.parametrize(
    "source",
    [
        "argument",
        "session-name",
        "vscode",
        "argument-beats-session-name",
        "session-name-beats-vscode",
    ],
)
def test_the_notebook_path_comes_from_the_argument_then_the_session_then_vscode(
    tmp_path: Path, source: str
):
    """The path is the line argument, else `JPY_SESSION_NAME`, else `__vsc_ipynb_file__`.

    Args:
        tmp_path: The test's own temporary directory.
        source: Which sources name the notebook; a path to no notebook stands in for each source
            that must lose.
    """
    right, wrong = tmp_path / "campaign.ipynb", tmp_path / "wrong.ipynb"
    vscode = {"vscode": right, "session-name-beats-vscode": wrong}.get(source)
    session = {"session-name": right, "argument-beats-session-name": wrong}.get(
        source, right if source == "session-name-beats-vscode" else None
    )
    argument = f" {right}" if source.startswith("argument") else ""
    cells = [_code(LOAD)]
    if vscode is not None:
        cells.append(_code(f"__vsc_ipynb_file__ = {str(vscode)!r}", active="ipynb"))
    body = _marker_test("test_here", tmp_path / "here")
    _write_notebook(tmp_path, [*cells, _magic(body, argument)])
    _run(tmp_path, [*cells, _magic(body, argument)], _env(session=session))

    assert (tmp_path / "here").read_text() == "ran"


def test_a_failing_test_makes_the_cell_raise(tmp_path: Path):
    """A failing test raises from the cell, after the cell's other tests have run.

    The cells after the failing one do not run.

    Args:
        tmp_path: The test's own temporary directory.
    """
    body = (
        _marker_test("test_passes", tmp_path / "passed") + "\n\ndef test_fails():\n    assert False"
    )
    cells = [
        _code(LOAD),
        _magic(body),
        _code(
            f"from pathlib import Path\nPath({str(tmp_path / 'after')!r}).write_text('ran')",
            active="ipynb",
        ),
    ]
    path = _write_notebook(tmp_path, cells)
    _raised(tmp_path, cells, _env(session=path))

    assert (tmp_path / "passed").read_text() == "ran"
    assert not (tmp_path / "after").exists()


def _summary(output: str) -> dict:
    """Parse the line `SUMMARY` printed by `SUMMARY`'s cell.

    Args:
        output: The cell's standard output.

    Returns:
        The printed record.
    """
    (line,) = [line for line in output.splitlines() if line.startswith("SUMMARY ")]
    return json.loads(line.removeprefix("SUMMARY "))


def _datasets(run_file: h5py.File) -> dict[str, h5py.Dataset]:
    """Collect every dataset in a run file by its path.

    Args:
        run_file: The open run file.

    Returns:
        Dataset path to dataset.
    """
    found: dict[str, h5py.Dataset] = {}
    run_file.visititems(
        lambda name, node: found.update({name: node}) if isinstance(node, h5py.Dataset) else None
    )
    return found


def _expected_runs(directory: Path) -> list[dict]:
    """Read each run file in `directory` with h5py, in case order.

    Args:
        directory: The test's run directory.

    Returns:
        One mapping per file: its `case`, and the `units` and `sums` of every dataset by path.
    """
    runs = []
    for path in sorted(directory.glob("run_*.h5")):
        with h5py.File(path, "r") as run_file:
            datasets = _datasets(run_file)
            runs.append(
                {
                    "case": int(run_file.attrs["case"]),
                    "units": {name: node.attrs["units"] for name, node in datasets.items()},
                    "sums": {name: float(node[:].sum()) for name, node in datasets.items()},
                }
            )
    return runs


def test_a_real_campaign_runs_in_workers_and_load_runs_reads_it_back(tmp_path: Path):
    """A `leo500` campaign of four cases runs across workers; `load_runs` reads every run back.

    One kernel runs the campaign and loads it with the test function and with its node path, in
    case order, matching what h5py reads. A second kernel that never ran it loads the same
    records. A third loads them, has the `.py` gain a `sim_out` mark, and a second `load_runs`
    there names the new folder: it read the file as it is on disk, not a cached module.

    Args:
        tmp_path: The test's own temporary directory.
    """
    by_function = SUMMARY.replace("CALL", "test_leo")
    by_name = SUMMARY.replace("CALL", '"test_leo"')
    campaign = [_code(LOAD), nbformat.v4.new_code_cell(CAMPAIGN.strip("\n"))]
    cells = [*campaign, _code(by_function, active="ipynb"), _code(by_name, active="ipynb")]
    path = _write_notebook(tmp_path, cells)
    env = _env(session=path)
    outputs = _run(tmp_path, cells, env)
    first, named = _summary(outputs[2]), _summary(outputs[3])
    fresh = _summary(_run(tmp_path, [_code(LOAD), _code(by_name, active="ipynb")], env)[1])

    expected = _expected_runs(tmp_path / "monte_carlo" / "campaign" / "test_leo")
    assert [run["case"] for run in expected] == [0, 1, 2, 3]
    assert len(set(first["worker_pids"])) > 1
    assert first["kernel_pid"] not in first["worker_pids"]
    for summary in (first, named, fresh):
        assert summary["cases"] == [0, 1, 2, 3]
        assert summary["units"] == [run["units"] for run in expected]
        assert summary["sums"] == [run["sums"] for run in expected]
    assert fresh["kernel_pid"] not in (first["kernel_pid"], named["kernel_pid"])

    elsewhere = tmp_path / "elsewhere"
    edit = f"""
        from pathlib import Path

        py = Path({str(path.with_suffix(".py"))!r})
        py.write_text(py.read_text().replace(
            "@pytest.mark.sim_runs(4)",
            "@pytest.mark.sim_runs(4)\\n@pytest.mark.sim_out({str(elsewhere)!r})",
        ))
    """
    probe = """
        from wise_dynamics.notebook import load_runs

        try:
            load_runs("test_leo")
        except FileNotFoundError as error:
            print("MISSING", error)
    """
    edited = _run(
        tmp_path,
        [
            _code(LOAD),
            _code(by_name, active="ipynb"),
            _code(edit, active="ipynb"),
            _code(probe, active="ipynb"),
        ],
        env,
    )
    assert _summary(edited[1])["cases"] == [0, 1, 2, 3]
    assert str(elsewhere / "campaign" / "test_leo") in edited[3]


def test_load_runs_with_no_run_files_names_the_expected_folder(tmp_path: Path):
    """With no run files, `load_runs` raises `FileNotFoundError` naming the folder it looked in.

    A test function, a method and a node-path string for each reach the same folders.

    Args:
        tmp_path: The test's own temporary directory.
    """
    plain = """
        class TestA:
            def test_x(self):
                pass

        def test_x():
            pass
    """
    probe = """
        from wise_dynamics.notebook import load_runs

        for test in (test_x, TestA.test_x, "test_x", "TestA::test_x"):
            try:
                load_runs(test)
            except FileNotFoundError as error:
                print("MISSING", error)
    """
    cells = [_code(LOAD), _code(plain), _code(probe, active="ipynb")]
    path = _write_notebook(tmp_path, cells)
    _sync(path)
    output = _run(tmp_path, cells, _env(session=path))[2]

    base = tmp_path / "monte_carlo" / "campaign"
    missing = [line for line in output.splitlines() if line.startswith("MISSING ")]
    assert len(missing) == 4
    for line, folder in zip(
        missing,
        [base / "test_x", base / "TestA" / "test_x", base / "test_x", base / "TestA" / "test_x"],
        strict=True,
    ):
        assert str(folder) in line


def test_load_runs_names_a_test_that_does_not_collect(tmp_path: Path):
    """A node that is not in the `.py` raises an error naming it, not a missing-folder error.

    Args:
        tmp_path: The test's own temporary directory.
    """
    probe = """
        from wise_dynamics.notebook import load_runs

        try:
            load_runs("test_absent")
        except Exception as error:
            print("RAISED", type(error).__name__, error)
    """
    cells = [_code(LOAD), _code("def test_present():\n    pass"), _code(probe, active="ipynb")]
    path = _write_notebook(tmp_path, cells)
    _sync(path)
    output = _run(tmp_path, cells, _env(session=path))[2]

    assert "RAISED" in output
    assert "test_absent" in output
    assert "FileNotFoundError" not in output
