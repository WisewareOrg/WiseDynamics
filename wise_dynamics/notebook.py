"""Monte Carlo campaigns from a jupytext-paired notebook: the `%%monte_carlo` magic and `load_runs`.

`%load_ext wise_dynamics.notebook` registers the magic. It needs the `notebook` extra of
`wise-dynamics`. `wise_dynamics/basilisk/README.md` documents both.
"""

import ast
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

try:
    import jupytext
    import jupytext.cli
    from IPython import get_ipython
    from IPython.core.interactiveshell import InteractiveShell
    from jupytext.formats import format_name_for_ext
except ImportError as error:
    raise ImportError(
        "wise_dynamics.notebook needs IPython and jupytext; "
        "install them with the extra: pip install 'wise-dynamics[notebook]'"
    ) from error

from wise_dynamics.pytest_plugin import _run_dir

__all__ = ["load_ipython_extension", "load_runs"]

_REPORT = "WISE_DYNAMICS_RUN_DIR_REPORT"
_MAGIC = "%%monte_carlo"
_PAIR_HOW = "pair it with `jupytext --set-formats ipynb,py:percent <notebook>`"


def load_ipython_extension(ipython: InteractiveShell) -> None:
    """Register the `%%monte_carlo` cell magic.

    Args:
        ipython: The interactive shell loading the extension.
    """

    def monte_carlo(line: str, cell: str) -> None:
        """Run the cell's tests as a Monte Carlo campaign.

        Args:
            line: The rest of the magic line, optionally the notebook's path.
            cell: The cell's source after the magic line.

        Raises:
            RuntimeError: The cell cannot be run as a campaign, or a test or the cell fails; the
                message says what to do.
        """
        notebook = _notebook_path(line, ipython.user_ns)
        if _normalized(cell) not in _saved_magic_bodies(notebook):
            raise RuntimeError(
                "This cell has unsaved changes. Save the notebook and run the cell again."
            )
        py = _paired_py(notebook)
        _sync_py(notebook, py)
        tests = _test_names(ipython.input_transformer_manager.transform_cell(cell))
        if not tests:
            raise RuntimeError("This cell defines no test.")
        ipython.run_cell(cell).raise_error()
        exit_code = pytest.main(_pytest_args(py, tests))
        if exit_code != pytest.ExitCode.OK:
            raise RuntimeError(f"pytest exited with {exit_code!r}")

    ipython.register_magic_function(monte_carlo, "cell", "monte_carlo")


def load_runs(
    test: str | Callable[..., object], notebook: str | os.PathLike[str] | None = None
) -> list[dict[str, Any]]:
    """Read back the run files a notebook test wrote.

    Args:
        test: The test function or method defined in the notebook, or its node path in the paired
            `.py`, such as `"test_x"` or `"TestA::test_x"`.
        notebook: The `.ipynb` or `.py`; by default the notebook the magic would use.

    Returns:
        One record per run file, sorted by `attrs["case"]`: `attrs` holds the root attributes and
        `datasets` maps each HDF5 dataset path to `(values, units)`.

    Raises:
        RuntimeError: The test does not collect from the notebook's `.py`, or the node collects
            tests of more than one function.
        FileNotFoundError: The test's run directory holds no run file.
    """
    node = test if isinstance(test, str) else test.__qualname__.replace(".", "::")
    path = _notebook_path(os.fspath(notebook) if notebook is not None else "", _user_ns())
    py = _paired_py(path) if path.suffix == ".ipynb" else path
    with tempfile.TemporaryDirectory() as scratch:
        report = Path(scratch) / "run_dir.json"
        collected = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                f"{py}::{node}",
                "--collect-only",
                "-q",
                "-o",
                "addopts=",
                "-p",
                "no:xdist",
                "-p",
                "wise_dynamics.notebook",
            ],
            cwd=py.parent,
            env={**os.environ, _REPORT: str(report)},
            capture_output=True,
            text=True,
        )
        if collected.returncode or not report.exists():
            raise RuntimeError(
                f"{py}::{node} does not collect:\n{collected.stdout}{collected.stderr}"
            )
        directories = json.loads(report.read_text())
    if len(directories) != 1:
        raise RuntimeError(
            f"{py}::{node} collects tests in {len(directories)} run directories; it must name a "
            "single test function or method"
        )
    directory = Path(directories[0])
    files = sorted(directory.glob("run_*.h5"))
    if not files:
        raise FileNotFoundError(f"no run files in {directory}")
    return sorted((_read_run(file) for file in files), key=lambda run: run["attrs"]["case"])


def pytest_collection_finish(session: pytest.Session) -> None:
    """Write the collected tests' run directories to the file `WISE_DYNAMICS_RUN_DIR_REPORT` names.

    Args:
        session: The finished collection session.
    """
    report = os.environ.get(_REPORT)
    if report and session.items:
        Path(report).write_text(json.dumps(sorted({str(_run_dir(item)) for item in session.items})))


def _user_ns() -> dict[str, Any]:
    """Return the running shell's namespace.

    Returns:
        The user namespace, empty outside IPython.
    """
    shell = get_ipython()
    return {} if shell is None else shell.user_ns


def _notebook_path(line: str, user_ns: dict[str, Any]) -> Path:
    """Find the notebook: the argument, else `JPY_SESSION_NAME`, else `__vsc_ipynb_file__`.

    Args:
        line: The argument, possibly empty.
        user_ns: The user namespace, which VS Code gives `__vsc_ipynb_file__`.

    Returns:
        The notebook's path.

    Raises:
        RuntimeError: None of the three names a notebook.
    """
    found = line.strip() or os.environ.get("JPY_SESSION_NAME") or user_ns.get("__vsc_ipynb_file__")
    if not found:
        raise RuntimeError(
            "The notebook's path is unknown; pass it as an argument, such as "
            "`%%monte_carlo path/to/notebook.ipynb`."
        )
    return Path(found)


def _normalized(source: str) -> str:
    r"""Normalize a cell body for comparison.

    Args:
        source: The cell body.

    Returns:
        The body with `\r\n` as `\n`, trailing whitespace stripped per line, and leading and
        trailing blank lines dropped.
    """
    lines = [line.rstrip() for line in source.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip("\n")


def _saved_magic_bodies(notebook: Path) -> list[str]:
    """Collect the bodies of the saved `%%monte_carlo` cells.

    Args:
        notebook: The `.ipynb` on disk.

    Returns:
        Each saved code cell's normalized source after its `%%monte_carlo` first line.
    """
    bodies = []
    for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
        source = cell["source"]
        text = source if isinstance(source, str) else "".join(source)
        magic, _, body = text.partition("\n")
        if cell["cell_type"] == "code" and magic.startswith(_MAGIC):
            bodies.append(_normalized(body))
    return bodies


def _jupytext(arguments: list[str]) -> tuple[int, str]:
    """Run jupytext's command line, capturing what it prints.

    Args:
        arguments: The command-line arguments.

    Returns:
        The exit status and the standard output and error, joined.
    """
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        status = jupytext.cli.jupytext(arguments)
    return status, out.getvalue() + err.getvalue()


def _paired_py(notebook: Path) -> Path:
    """Find the `.py` a notebook is paired with.

    Args:
        notebook: The `.ipynb`.

    Returns:
        The paired `.py`.

    Raises:
        RuntimeError: The notebook is not paired with a `.py`.
    """
    _, output = _jupytext(["--paired-paths", str(notebook)])
    pys = [line for line in output.splitlines() if line.endswith(".py")]
    if not pys:
        raise RuntimeError(f"{notebook} has no paired .py; {_PAIR_HOW}")
    return Path(pys[0])


def _pytest_args(py: Path, tests: list[str]) -> list[str]:
    """Build the arguments of the campaign's pytest run.

    Args:
        py: The paired `.py` holding the tests.
        tests: The names of the tests to run.

    Returns:
        One node id per test, `-n auto`, and `--ci` when the `CI` environment variable is set.
    """
    return [
        *(f"{py}::{name}" for name in tests),
        "-n",
        "auto",
        *(["--ci"] if os.environ.get("CI") else []),
    ]


def _sync_py(notebook: Path, py: Path) -> None:
    """Bring the paired `.py` up to date with the saved notebook, never over its own edits.

    Args:
        notebook: The `.ipynb` on disk.
        py: Its paired `.py`.

    Raises:
        RuntimeError: The `.py` differs from the notebook and is newer, or jupytext fails.
    """
    if _in_sync(notebook, py):
        return
    if py.exists() and py.stat().st_mtime > notebook.stat().st_mtime:
        raise RuntimeError(
            "The .py was edited after the notebook was saved; reload the notebook from disk "
            "before running"
        )
    status, output = _jupytext(["--sync", str(notebook)])
    if status:
        raise RuntimeError(f"jupytext could not sync {notebook}: {output}")


def _test_names(source: str) -> list[str]:
    """List the top-level tests a cell defines.

    Args:
        source: The cell's Python source.

    Returns:
        The names of its top-level `test*` functions and `Test*` classes.
    """
    return [
        node.name
        for node in ast.parse(source).body
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name.startswith("test")
        )
        or (isinstance(node, ast.ClassDef) and node.name.startswith("Test"))
    ]


def _in_sync(notebook: Path, py: Path) -> bool:
    """Say whether the `.py` already holds what jupytext would write from the saved notebook.

    Args:
        notebook: The `.ipynb` on disk.
        py: Its paired `.py`.

    Returns:
        True when `py` exists and its text equals the notebook rendered in the pair's format.
    """
    if not py.exists():
        return False
    saved = jupytext.read(notebook)
    fmt = {"extension": ".py", "format_name": format_name_for_ext(saved.metadata, ".py")}
    return jupytext.writes(saved, fmt=fmt) == py.read_text(encoding="utf-8")


def _read_run(path: Path) -> dict[str, Any]:
    """Read one run file.

    Args:
        path: The run file.

    Returns:
        Its root attributes and its datasets by path as `(values, units)`.
    """
    datasets: dict[str, tuple[np.ndarray, str]] = {}

    def collect(name: str, node: h5py.HLObject) -> None:
        """Record a dataset.

        Args:
            name: The object's path in the file.
            node: The object; only datasets are recorded.
        """
        if isinstance(node, h5py.Dataset):
            datasets[name] = (node[...], node.attrs["units"])

    with h5py.File(path, "r") as run_file:
        attrs = dict(run_file.attrs)
        run_file.visititems(collect)
    return {"attrs": attrs, "datasets": datasets}
