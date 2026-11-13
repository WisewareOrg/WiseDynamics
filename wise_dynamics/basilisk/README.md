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
| `wise_dynamics/pytest_plugin.py` | the pytest plugin's marks, `--full-monte-carlo`, `--sim-session` and `sim_*` fixtures |
| `wise_dynamics/sessions.py` | the session pointer and layout: `session_directory`, `current_session`, `recorded_tests` and `read_test_results` |
| `wise_dynamics/notebook.py` | the `%%monte_carlo` notebook magic and `load_runs` (the `notebook` extra) |
| `wise_dynamics/basilisk/vehicle.py` | `Vehicle`: the run file every vehicle writes |
| `wise_dynamics/basilisk/scenario.py` | `Scenario`: the simulation every scenario runs once and records |
| `wise_dynamics/basilisk/vehicles/` | one `Vehicle` subclass per module, the module named for the class; `VEHICLES` registers each under the module's name |
| `wise_dynamics/basilisk/scenarios/` | one `Scenario` subclass per module, named for the scenario |

Consumers import `Scenario`, `Vehicle`, `ImuSpacecraft`, `SCENARIOS` and `VEHICLES` from
`wise_dynamics.basilisk`, and a scenario class from its own module,
`wise_dynamics.basilisk.scenarios.<name>`. A test picks one scenario with the `sim_scenario` mark, a
name in `SCENARIOS`, and the scenario builds its own vehicle unless the `sim_vehicle` mark names
another.

## Vehicles

A vehicle is a `Vehicle` subclass: its parameters are keywords, each defaulting to the class
attribute of the same name, read when the vehicle is built, so a subclass that overrides an
attribute is built with its value. Its class attributes are its nominal values, and its class
docstring gives each parameter's units and frame. Every vehicle takes `noise_seed`, the seed of its
sensor noise. A vehicle class disperses itself: `dispersed(rng, *, noise_seed)` draws its own
dispersion from `rng`, with the 1-sigmas its class attributes give, and returns a vehicle built
with it and what it drew.

A vehicle lists the Basilisk modules it propagates in `models()` and its sensors in
`sensor_models(sample_period_s)`, which also sets each sensor's per-sample noise for that period: a
sensor's white noise is a density, and a sample carries 1-sigma density / sqrt(period). It hands out
sensor readers by role in `readers()`, and declares what it records in `recorded_fields()`: dataset
path to `(message, field, units)`. A vehicle may change what the recorders captured in
`finish(datasets)`, which `Scenario.run` calls on each vehicle after the run. `ImuSpacecraft`
records the attitude as `truth/q_BN` and turns Basilisk's MRP into it, one batch of quaternions W X
Y Z (Hamilton, body to N, W >= 0, no units), so an attitude leaves the simulation as a quaternion
and Basilisk's MRP stays inside it. `ImuSpacecraft.tracking()` returns a function giving the hub's
`q_BN`, `r_N` and `v_N` as of the last write, the ground's stand-in for tracking.
`write_run_file(path, datasets, attrs=None)`, from `Vehicle`, writes one HDF5 file, each dataset
float64, or UTF-8 text for a string array, with a `units` attribute.

## Scenarios

To add a scenario, add one module to `wise_dynamics/basilisk/scenarios/`, named for the scenario and
defining one `Scenario` subclass; there is nothing else to edit. `SCENARIOS` registers it under the
module's name. A module defining none or several raises `TypeError` when the package is imported.

A scenario is a `Scenario` subclass whose class attributes are its environment and its placement,
with units and frames in its class docstring. Its constructor takes no required argument and one
optional argument, the vehicle class to build in place of its own, which the `sim_vehicle` mark
supplies; a scenario that does not accept it cannot be used with that mark. It builds its vehicle,
places it, and holds it as `scenario.vehicle` and in `scenario.vehicles`. `disperse(rng, *,
noise_seed)` draws the scenario's placement from `rng`, as its docstring lists it, then has its
vehicle class draw its own dispersion from the same `rng` and build a vehicle with `noise_seed`,
places that vehicle, and records both sets of draws in `scenario.draws`. The new vehicle replaces
`scenario.vehicle`, so readers and other handles on it are taken after dispersing. `initial_state()`
returns the placement the vehicle starts from.

`scenario.fidelity`, an attribute from `Scenario` that defaults to `"lo"`, is the integrator `run`
gives every vehicle's spacecraft. A fidelity is the accuracy of the truth it propagates: each has a
budget on the truth position's distance from `hi`'s over one orbit, which `tests/test_fidelity.py`
holds every scenario to. The step is not a fidelity's: every fidelity steps at the largest step of
at most 1 s that divides the flight software's period. `run` refuses, with a `ValueError` before
anything runs, a period no step of at least 1 ms divides.

| fidelity | integrator | budget over one orbit |
|---|---|---|
| `hi` | RKF78, relative tolerance 1e-10, absolute 1e-12 | (reference) |
| `med` | RK4 | 1 mm |
| `lo` | Ralston's RK3 | 1 m |

[ADR 0003 rev 2](../../docs/decisions/0003-fidelity-tiers.md) records why, with the measurements.

`run(fsw_module, fsw_fields, *, duration_s, ground=None)`, from `Scenario`, builds the simulation
from `vehicles` and `fidelity`, and runs once. The vehicles' `models()` run every step. Their
`sensor_models` run in their own task at the flight software's `period_s`, an attribute it declares
in seconds, after the models at each instant they share, so they sample that instant's state. The
flight-software module runs in its own task at the same period, after the sensors, so it reads that
instant's sensor output. The recorders run at every dynamics step, after the rest, so truth and
flight output are recorded from the same instant. `fsw_fields` maps a dataset path to a
flight-software `(message, field, units)`, as `recorded_fields()` does for a vehicle, and may be
empty. `run` returns dataset path to `(array, units)`: `t_s` (s) first, then the vehicle's fields,
then `fsw_fields`.

A `ground` is any Basilisk module declaring `period_s`. It runs from t = 0 in a process of its own,
`ground_process`, whose priority is below the scenario process's, so at each instant they share it
runs after every task of the scenario process: after the flight software, and after the recorders.

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
| `sim_vehicle(name)` | the vehicle's name in `VEHICLES`, a module name in `wise_dynamics/basilisk/vehicles/`; the scenario builds that class instead of its own | the scenario's own |
| `sim_fidelity(level)` | `hi`, `med` or `lo`, as the Scenarios section defines them | `lo` |

A run is capped unless `--full-monte-carlo` is given: every test's case count is at most 2, and a
test marked for fewer runs its own. Bare `pytest`, the editor's test explorer and CI are therefore
capped, and a full campaign is asked for. A capped run writes to the `scratch` session whatever the
pointer says, and never changes the pointer. `WISE_DYNAMICS_CAPPED`, when non-empty in the
environment, forces a capped run even with `--full-monte-carlo`; a consumer's tests that run a
notebook set it, and an interactive kernel never does. The `sim_out` ini setting, a path relative to
the rootdir or an absolute one and not empty, is the root of every session in a project; default
`monte_carlo`.

`--sim-session <name>` makes `<name>` the active session and runs no test; with no name it prints
the active session and lists the others. A name starts with a letter or digit, then letters,
digits, `.`, `_` or `-`, and is neither `SESSION` nor `scratch`, so `--sim-session scratch` is
rejected. `just session [name]` wraps it. A Monte Carlo test run with no active session shows one
error naming `--sim-session`; the other tests of its file run. When `WISE_DYNAMICS_SESSION_REPORT`
names a file, `--sim-session` with no name also writes the active session's directory there as
JSON, `null` when none is active; `current_session()` reads it.

| fixture | gives |
|---|---|
| `sim_case_index` | the case number; a test using it, directly or through the fixtures below, runs once per case |
| `sim_duration_s` | the test's `sim_duration_s` |
| `sim_test_directory` | the test's directory in the session its runs land in (`scratch` for a capped run), the directory `sim_run_file` writes to |
| `sim_scenario` | the case's scenario at the test's `sim_fidelity`, dispersed from the case's two streams; its vehicle is `sim_scenario.vehicle`, and what it drew is `sim_scenario.draws` |
| `sim_run_file` | `sim_run_file(datasets, attrs=None)` writes `run_<NNNN>.h5` in the test's directory of the session its runs land in (`scratch` for a capped run) through `sim_scenario.vehicle` and returns its path, with root attributes `case`, `seed` (0), `fidelity`, `created_utc`, `versions_bsk`, `versions_wise_dynamics`, the install's provenance (`versions_wise_dynamics_install`, `pinned` or `local`; `versions_wise_dynamics_commit`; `versions_wise_dynamics_requested_revision`, empty for a local install; `versions_wise_dynamics_dirty`, always false for a pin) and the calling repository's provenance (`versions_caller_commit`, the commit of the git repository that holds the rootdir, and `versions_caller_dirty`, whether its tree has uncommitted changes; both absent when the rootdir is not in a git checkout), `draw_<name>` per entry of `sim_scenario.draws`, and `attrs` beside them |

Case `k` takes its draws and its noise seed from the two streams of
`numpy.random.SeedSequence(0, spawn_key=(k,)).spawn(2)`, so it is the same whatever `sim_runs`
is and reruns alone by its node id: `pytest "<test file>::test_case[k]" --full-monte-carlo`.

A session is a named working set holding one set of run files per test
([ADR 0002 rev 2](../../docs/decisions/0002-monte-carlo-sessions.md)). `<root>/SESSION` holds the
active session's name; the plugin and `load_runs` read it, so every way of running or reading
cases agrees on where runs are, and a running notebook kernel follows a switch. Outside pytest,
`wise_dynamics.sessions.current_session()` returns the directory of the session a run at the
current directory reads and writes, `scratch` included when `WISE_DYNAMICS_CAPPED` is set, and
raises `NoSessionError` when none is active. One run resolves
the session once, so switching it midway does not split the run. A test's run files are in
`<root>/<session>/<test file>/<class>/<function>/`: the test file is its path relative to the
rootdir without `.py`, and each enclosing class, outermost first, is a directory. Before the first
case runs, each worker writes `planned.json` into that directory,
`{"runs": N, "nodeid": "<test file>.py::<class>::<function>"}`: the test's case count after the
cap, and its node id without parameters. `wise_dynamics.sessions.recorded_tests(session)` lists a session's test folders with
their node ids, so a consumer finds a test's folder by its node id instead of parsing the path.

A case that starts deletes its own previous `run_<NNNN>.h5` and `run_<NNNN>.crash.json`, so
rerunning one case by its node id replaces only its result. A campaign of N cases also deletes, when
it is collected, its test's run files and crash records numbered N and above, so the leftovers of an
earlier, larger campaign are cleared; nothing else is. A case
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
`period_s` = 1.0, so it, the IMU and the dynamics all step at 1 Hz. Its
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
paired `.py` and runs the cell's top-level `test*` functions and `Test*` classes through pytest with
`-n auto` and `--full-monte-carlo` (capped, without the option, when `WISE_DYNAMICS_CAPPED` is set).
The tests also include any test function or `Test*` class the cell's code creates, such as by
assigning it into the namespace. The magic runs the cell first and then flies the tests it defines
or creates, so a cell that defines or creates no test is refused after it has run. The cell then
raises if any test fails, so Run All stops before the plot cells. The notebook's path comes from the
argument, else `__vsc_ipynb_file__` (VS Code), else `JPY_SESSION_NAME` (a Jupyter server).

Save the notebook first; a cell that differs from the notebook on disk is refused. The notebook
may be the `.ipynb` or the paired `.py` opened as a notebook. Every worker imports the `.py`, so
analysis and plot cells must be notebook-only (`active="ipynb"`). In an `.ipynb` the magic marks
each code cell after the last `%%monte_carlo` cell that has no `active` metadata, saves the
notebook and re-syncs before running. Mark a cell above the last `%%monte_carlo` cell, such as the
plots of an earlier campaign, or any cell of a `.py` opened as a notebook, yourself.

`load_runs(test, notebook=None)` reads the test's run files in the active session back, finished or
not, and prints `<session>: <found> of <planned> cases`, followed by `, crashed: <cases>` when any
case left a crash record. It returns one record per run file, sorted by case: `attrs` holds the root
attributes and `datasets` maps each dataset path to `(values, units)`, a string dataset's values as
`str`. `test` is the test function or method, or its node path such as `"TestA::test_x"`. It works
from a restarted kernel, reads the pointer on every call, raises `RuntimeError` naming
`--sim-session` when no session is active, and raises `FileNotFoundError` naming the folder when the
test has no results in the session.