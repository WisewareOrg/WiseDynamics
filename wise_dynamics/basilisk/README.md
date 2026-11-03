# wise_dynamics.basilisk

Basilisk vehicles, the scenarios that place them, and the pytest plugin whose session over
`--sim-runs` cases is a Monte Carlo run. A vehicle is mass properties, sensors and logging, and
knows nothing about where it flies. A scenario is a prebuilt operating environment and the placement
of the vehicles added to it, and runs them once with a flight-software module. Basilisk is the
simulation host by ADR 0001 rev 1. Each symbol's docstring carries its arguments, returns and
raises.

## Modules

| module | holds |
|---|---|
| `wise_dynamics/pytest_plugin.py` | the pytest plugin's options and `sim_*` fixtures |
| `wise_dynamics/basilisk/vehicle.py` | `Vehicle`: the interface every vehicle meets, its draws and its run file |
| `wise_dynamics/basilisk/scenario.py` | `Scenario`: the interface every scenario meets, and its one run |
| `wise_dynamics/basilisk/vehicles/` | one `Vehicle` subclass per module, the module named for the class; `VEHICLES` in its `__init__.py` maps each name to its class |
| `wise_dynamics/basilisk/scenarios/` | one `Scenario` subclass per module, the module named for the class; `SCENARIOS` in its `__init__.py` maps each name to its class |

Consumers import from `wise_dynamics.basilisk`. A session picks one vehicle with `--sim-vehicle`
and one scenario with `--sim-scenario`, each a name in its registry.

## Vehicles

A vehicle is a `Vehicle` subclass: its parameters are keywords, each defaulting to the class
attribute of the same name, read when the vehicle is built, so a subclass that overrides an
attribute is built with its value. Its class attributes are its nominal values, and its class
docstring gives each parameter's units and frame. `standard_dispersion` maps a parameter to its
per-axis 1-sigma, and `draw(rng)` returns that dispersion, in order, as nominal plus normal samples
from the numpy `Generator`. Every vehicle takes `noise_seed`, which the plugin sets per case.

A vehicle lists its Basilisk modules in `models()`, hands out sensor readers by role in
`readers()`, and declares what it records in `recorded_fields()`: dataset path to
`(message, field, units)`. `write_run_file(path, datasets, attrs=None)` writes one HDF5 file, each
dataset float64 with a `units` attribute.

## Scenarios

A scenario is a `Scenario` subclass whose class attributes are its environment and its
placement, with units and frames in its class docstring; its constructor takes no arguments.
`standard_dispersion` and `draw(rng)` are as for a vehicle, and `disperse(draws)` sets drawn
values on one instance, refusing any attribute outside the standard dispersion. A scenario takes
one vehicle: `add(vehicle)` places it at the scenario's placement, and a second `add` is refused.
`initial_state()` returns the placement the vehicle starts from, valid before `add` and `run`.

`run(fsw_module, *, fsw_fields, duration_s, record_s=10.0)` builds the simulation and runs once.
`fsw_fields` maps a dataset path to a flight-software `(message, field, units)`, as
`recorded_fields()` does for a vehicle, and may be empty. `run` returns dataset path to
`(array, units)`: `t_s` (s) first, then the vehicle's fields, then `fsw_fields`.

## pytest plugin

Installing `wise-dynamics` registers the plugin through its `pytest11` entry point, so a
consumer's `conftest.py` names nothing.

| option | default | meaning |
|---|---|---|
| `--sim-runs` | 8 | cases in the session, at least 1 |
| `--sim-out` | a new directory under the system temporary directory | where run files go; a directory already holding run files is refused |
| `--sim-seed` | 0 | session seed |
| `--sim-duration-s` | 5700 | simulated seconds per case |
| `--sim-scenario` | `leo500` | the scenario's name in `SCENARIOS` |
| `--sim-vehicle` | `imu_spacecraft` | the vehicle's name in `VEHICLES` |

| fixture | gives |
|---|---|
| `sim_case_index` | the case number; a test using it, directly or through the fixtures below, runs once per case |
| `sim_duration_s` | `--sim-duration-s` |
| `sim_draws` | the case's draws, name to value: by default the scenario's `draw`, then the vehicle's; override it to supply your own |
| `sim_vehicle` | the case's vehicle, built with the `sim_draws` entries outside its scenario's standard dispersion and its noise seeded from the case |
| `sim_scenario` | the case's scenario, dispersed with the `sim_draws` entries in its standard dispersion, holding `sim_vehicle` |
| `sim_run_file` | `sim_run_file(datasets, attrs=None)` writes `run_<NNNN>.h5` under `--sim-out` through the vehicle and returns its path, with root attributes `case`, `seed`, `created_utc`, `versions_bsk`, `versions_wise_dynamics` and `draw_<name>` per draw, and `attrs` beside them |

Case `k` of seed `s` takes its draws and its noise seed from the two streams of
`numpy.random.SeedSequence(s, spawn_key=(k,)).spawn(2)`, so it is the same whatever `--sim-runs`
is and reruns alone: `pytest <test file> --sim-seed S --sim-runs N "<test file>::test_case[k]"`.

A case, with Basilisk's stock `inertial3D` guidance standing in for flight software that reads
`sim_vehicle.readers()` and starts from `sim_scenario.initial_state()`. Its output joins the run
file through `fsw_fields`. `pytest <test file> -n 8 --sim-runs 852 --sim-out DIR` runs it as a
Monte Carlo run; `-n` runs workers in parallel and needs `pytest-xdist`, which a consumer adds
itself:

```python
from Basilisk.fswAlgorithms import inertial3D


def test_case(sim_scenario, sim_duration_s, sim_run_file):
    guidance = inertial3D.inertial3D()
    guidance.ModelTag = "guidance"
    fsw_fields = {"guidance/sigma_RN": (guidance.attRefOutMsg, "sigma_RN", "MRP")}
    datasets = sim_scenario.run(
        guidance, fsw_fields=fsw_fields, duration_s=sim_duration_s, record_s=10.0
    )
    sim_run_file(datasets, {"versions_guidance": "1.0"})
```
