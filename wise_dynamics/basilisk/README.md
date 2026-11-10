# wise_dynamics.basilisk

Basilisk vehicles, the scenarios that place them, and the pytest plugin whose tests are
Monte Carlo campaigns. A vehicle is mass properties, sensors and logging, and knows nothing about
where it flies. A scenario is a prebuilt operating environment and the placement of the vehicle it
builds, and runs them once with a flight-software module. Basilisk is the
simulation host by ADR 0001 rev 1. Each symbol's docstring carries its arguments, returns and
raises.

## Modules

| module | holds |
|---|---|
| `wise_dynamics/pytest_plugin.py` | the pytest plugin's marks, `--ci`, `--sim-session` and `sim_*` fixtures |
| `wise_dynamics/sessions.py` | the session pointer and layout: `session_directory`, `recorded_tests` and `read_test_results` |
| `wise_dynamics/notebook.py` | the `%%monte_carlo` notebook magic and `load_runs` (the `notebook` extra) |
| `wise_dynamics/basilisk/vehicle.py` | `Vehicle`: the run file every vehicle writes |
| `wise_dynamics/basilisk/scenario.py` | `Scenario`: the simulation every scenario runs once and records |
| `wise_dynamics/basilisk/vehicles/` | one `Vehicle` subclass per module, the module named for the class |
| `wise_dynamics/basilisk/scenarios/` | one `Scenario` subclass per module, named for the scenario |

Consumers import `Scenario`, `Vehicle`, `ImuSpacecraft` and `SCENARIOS` from
`wise_dynamics.basilisk`, and a scenario class from its own module,
`wise_dynamics.basilisk.scenarios.<name>`. A test picks one scenario with the `sim_scenario` mark, a
name in `SCENARIOS`, and the scenario builds its own vehicle.

## Vehicles

A vehicle is a `Vehicle` subclass: its parameters are keywords, each defaulting to the class
attribute of the same name, read when the vehicle is built, so a subclass that overrides an
attribute is built with its value. Its class attributes are its nominal values, and its class
docstring gives each parameter's units and frame. Every vehicle takes `noise_seed`, the seed of its
sensor noise. A vehicle class disperses itself: `dispersed(rng, *, noise_seed)` draws its own
dispersion from `rng`, with the 1-sigmas its class attributes give, and returns a vehicle built
with it and what it drew.

A vehicle lists its Basilisk modules in `models()`, hands out sensor readers by role in
`readers()`, and declares what it records in `recorded_fields()`: dataset path to
`(message, field, units)`. `write_run_file(path, datasets, attrs=None)`, from `Vehicle`, writes one
HDF5 file, each dataset float64 with a `units` attribute.

## Scenarios

To add a scenario, add one module to `wise_dynamics/basilisk/scenarios/`, named for the scenario and
defining one `Scenario` subclass; there is nothing else to edit. `SCENARIOS` registers it under the
module's name. A module defining none or several raises `TypeError` when the package is imported.

A scenario is a `Scenario` subclass whose class attributes are its environment, its placement and
its recording period `record_s`, with units and frames in its class docstring. Its constructor takes
no arguments: it builds its vehicle, places it, and holds it as `scenario.vehicle` and in
`scenario.vehicles`. `disperse(rng, *, noise_seed)` draws the scenario's placement from
`rng`, as its docstring lists it, then has its vehicle class draw its own dispersion from the same
`rng` and build a vehicle with `noise_seed`, places that vehicle, and records both sets of draws in
`scenario.draws`. The new vehicle replaces `scenario.vehicle`, so readers and other handles on it
are taken after dispersing. `initial_state()` returns the placement the vehicle starts from.

`scenario.fidelity`, an attribute from `Scenario` that defaults to `"lo"`, is the level `run`
builds at. `run` derives the integration step and the integrator it gives every vehicle's
spacecraft from it. The differences below are measured on Leo500 at the end of one 5700 s orbit,
against hi:

| fidelity | integrator | step | position | attitude |
|---|---|---|---|---|
| `hi` | RK4 | 0.02 s | (reference) | (reference) |
| `med` | RK4 | 0.05 s | 1.9 um | 5.9e-13 rad |
| `lo` | RK2 | 0.05 s | 17.7 cm | 9.05 urad |

`run(fsw_module, fsw_fields, *, duration_s)`, from `Scenario`, builds the simulation from
`vehicles`, `fidelity` and `record_s`, and runs once. The vehicles run every step of the fidelity.
The flight-software module runs in its own task at its own `period_s`, an attribute it declares in
seconds, after the vehicles at each instant they share, so it reads that instant's sensor output.
The recorders run every `record_s`, after both, so truth and flight output are recorded from the
same instant. `fsw_fields` maps a dataset path to a flight-software `(message, field, units)`, as
`recorded_fields()` does for a vehicle, and may be empty. `run` returns dataset path to `(array,
units)`: `t_s` (s) first, then the vehicle's fields, then `fsw_fields`.

## pytest plugin

Installing `wise-dynamics` registers the plugin through its `pytest11` entry point, so a
consumer's `conftest.py` names nothing.

Each test that uses a `sim_` fixture is one Monte Carlo campaign, configured by marks on the
function, a class or the module; the closest mark wins. Each mark takes exactly one argument, and a
bad one fails collection of the test, naming its node id and the mark.

| mark | argument | default |
|---|---|---|
| `sim_runs(n)` | cases in the campaign, an integer of at least 1 | 8 |
| `sim_duration_s(seconds)` | simulated seconds per case, positive | 5700 |
| `sim_scenario(name)` | the scenario's name in `SCENARIOS` | `leo500` |
| `sim_fidelity(level)` | `hi`, `med` or `lo`, as the Scenarios section defines them | `lo` |

`--ci` caps every test's case count at 2, and a test marked for fewer runs its own. The `sim_out`
ini setting, a path relative to the rootdir or an absolute one and not empty, is the root of every
session in a project; default `monte_carlo`.

`--sim-session <name>` makes `<name>` the active session and runs no test; with no name it prints
the active session and lists the others. A name starts with a letter or digit, then letters,
digits, `.`, `_` or `-`, and is not `SESSION`. `just session [name]` wraps it. A Monte Carlo test
run with no active session shows one error naming `--sim-session`; the other tests of its file run.

| fixture | gives |
|---|---|
| `sim_case_index` | the case number; a test using it, directly or through the fixtures below, runs once per case |
| `sim_duration_s` | the test's `sim_duration_s` |
| `sim_scenario` | the case's scenario at the test's `sim_fidelity`, dispersed from the case's two streams; its vehicle is `sim_scenario.vehicle`, and what it drew is `sim_scenario.draws` |
| `sim_run_file` | `sim_run_file(datasets, attrs=None)` writes `run_<NNNN>.h5` in the test's directory of the active session through `sim_scenario.vehicle` and returns its path, with root attributes `case`, `seed` (0), `fidelity`, `created_utc`, `versions_bsk`, `versions_wise_dynamics`, the install's provenance (`versions_wise_dynamics_install`, `pinned` or `local`; `versions_wise_dynamics_commit`; `versions_wise_dynamics_requested_revision`, empty for a local install; `versions_wise_dynamics_dirty`, always false for a pin) and `draw_<name>` per entry of `sim_scenario.draws`, and `attrs` beside them |

Case `k` takes its draws and its noise seed from the two streams of
`numpy.random.SeedSequence(0, spawn_key=(k,)).spawn(2)`, so it is the same whatever `sim_runs`
is and reruns alone by its node id: `pytest "<test file>::test_case[k]"`.

A session is a named working set holding one set of run files per test
([ADR 0002 rev 1](../../docs/decisions/0002-monte-carlo-sessions.md)). `<root>/SESSION` holds the
active session's name; the plugin and `load_runs` read it, so every way of running or reading
cases agrees on where runs are, and a running notebook kernel follows a switch. One run resolves
the session once, so switching it midway does not split the run. A test's run files are in
`<root>/<session>/<test file>/<class>/<function>/`: the test file is its path relative to the
rootdir without `.py`, and each enclosing class, outermost first, is a directory. Before the first
case runs, each worker writes `planned.json` into that directory,
`{"runs": N, "nodeid": "<test file>.py::<class>::<function>"}`: the test's case count after the
`--ci` cap, and its node id without parameters. `wise_dynamics.sessions.recorded_tests(session)` lists a session's test folders with
their node ids, so a consumer finds a test's folder by its node id instead of parsing the path.

A case that starts deletes its own previous `run_<NNNN>.h5` and `run_<NNNN>.crash.json`, and
nothing else is cleared, so rerunning one case by its node id replaces only its result. A case
that fails in setup or call without having written its run file leaves `run_<NNNN>.crash.json`,
`{"exception": ..., "traceback": ...}`; so does a case whose pytest-xdist worker dies, such as a
segfault. A case that wrote its run file and then failed keeps the run file, unless its worker
died: then the crash record replaces the run file.
`wise_dynamics.sessions.read_test_results(directory)` returns `planned`, the `runs` and the
`crashes` of cases `0` to `planned-1`, ignoring any higher-numbered file left after `sim_runs`
dropped. Edge outcomes: Ctrl-C leaves no record, so the case is missing; a skipped case keeps its
previous result; an xfail-marked case that raises leaves no record; a worker that dies in a case
whose parameter id is not a plain case number leaves no record.

`--collect-only` writes nothing. The terminal summary names each directory that received run
files. An install that is neither a git pin (`uv sync` against a git dependency) nor an editable
local checkout (`uv pip install -e <checkout>`) stops the pytest run before collection.

A case, with Basilisk's stock `inertial3D` guidance standing in for flight software that reads
`sim_scenario.vehicle.readers()` and starts from `sim_scenario.initial_state()`. It declares
`period_s` = 1.0, so it steps at 1 Hz while the default fidelity steps the vehicle at 20 Hz. Its
output joins the run file through `fsw_fields`.
Pick a session with `pytest --sim-session <name>`, then `pytest <test file> -n 8` runs it as a
Monte Carlo campaign; `-n` runs workers in parallel and needs `pytest-xdist`, which a consumer
adds itself:

```python
import pytest
from Basilisk.fswAlgorithms import inertial3D


class Guidance(inertial3D.inertial3D):
    period_s = 1.0


@pytest.mark.sim_runs(2)
@pytest.mark.sim_duration_s(20)
def test_case(sim_scenario, sim_duration_s, sim_run_file):
    guidance = Guidance()
    guidance.ModelTag = "guidance"
    fsw_fields = {"guidance/sigma_RN": (guidance.attRefOutMsg, "sigma_RN", "MRP")}
    datasets = sim_scenario.run(guidance, fsw_fields, duration_s=sim_duration_s)
    sim_run_file(datasets, {"versions_guidance": "1.0"})
```

## Notebooks

A jupytext-paired notebook (`jupytext --set-formats ipynb,py:percent <notebook>`) runs a campaign
from a cell. Install the `notebook` extra, `wise-dynamics[notebook]`, and load the extension with
`%load_ext wise_dynamics.notebook`. `%%monte_carlo [notebook path]` as a cell's first line syncs the
paired `.py` and runs the cell's top-level `test*` functions and `Test*` classes through pytest
with `-n auto` (plus `--ci` when the `CI` environment variable is set); the cell then raises if any
test fails, so Run All stops before the plot cells. The notebook's path comes from the argument,
else `__vsc_ipynb_file__` (VS Code), else `JPY_SESSION_NAME` (a Jupyter server).

Save the notebook first; a cell that differs from the notebook on disk is refused. The notebook
may be the `.ipynb` or the paired `.py` opened as a notebook. Every worker imports the `.py`, so
analysis and plot cells must be notebook-only (`active="ipynb"`). In an `.ipynb` the magic marks
each code cell after the last `%%monte_carlo` cell that has no `active` metadata, saves the
notebook and re-syncs before running. Mark a cell above the last `%%monte_carlo` cell, such as the
plots of an earlier campaign, or any cell of a `.py` opened as a notebook, yourself.

`load_runs(test, notebook=None)` reads the test's run files in the active session back, finished or
not, and prints `<session>: <found> of <planned> cases`, followed by `, crashed: <cases>` when any
case left a crash record. It returns one record per run file, sorted by case: `attrs` holds the
root attributes and `datasets` maps each dataset path to `(values, units)`. `test` is the test
function or method, or its node path such as `"TestA::test_x"`. It works from a restarted kernel,
reads the pointer on every call, raises `RuntimeError` naming `--sim-session` when no session is
active, and raises `FileNotFoundError` naming the folder when the test has no results in the
session.
