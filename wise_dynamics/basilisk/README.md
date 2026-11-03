# wise_dynamics.basilisk

Basilisk spacecraft plants, the one-case runner, the run-file writer, and the pytest plugin
whose session over `--sim-runs` cases is a Monte Carlo run. Basilisk is the first
simulation host by ADR 0001 rev 1. This file is the reference for the `wise-dynamics`
distribution; each symbol's docstring carries its arguments, returns and raises.

## Modules

| module | holds |
|---|---|
| `wise_dynamics/dispersion.py` | `draw_standard`: a plant's standard per-case dispersion, as constructor keywords |
| `wise_dynamics/run_file.py` | `write_run`: one HDF5 run file |
| `wise_dynamics/session_options.py` | `SessionOptions`: a pytest session's parsed `--sim-*` options, internal to the plugin |
| `wise_dynamics/pytest_plugin.py` | the pytest plugin's hooks and `sim_*` fixtures |
| `wise_dynamics/basilisk/plant.py` | `Plant`: the interface every plant meets |
| `wise_dynamics/basilisk/runner.py` | `run_case`: runs one case |
| `wise_dynamics/basilisk/plants/__init__.py` | `PLANTS`: the registry of plants by name |
| `wise_dynamics/basilisk/plants/imu_spacecraft.py` | `ImuSpacecraft`: the IMU spacecraft plant |

Consumers import from the two packages, `wise_dynamics` and `wise_dynamics.basilisk`; each
declares its public names in `__all__`, and those import paths stay the same whichever module a
name lives in.

## Plants

A plant is a `Plant` subclass that takes its parameters as constructor keywords, with
documented defaults and units in their names, and keeps each as an attribute. It owns one Basilisk
simulation, runs once through `execute(duration_s)`, and provides `readers()`,
`recorded_fields()`, `initial_state()`, and the class attributes `mu` (m^3/s^2) and
`standard_dispersion`.

Each reader `readers()` returns holds its plant, so it reads the plant's live output for as long
as it is held. The messages in `recorded_fields()`, `spacecraft` and `imu`, along with anything
reached through them such as `spacecraft.gravField.gravBodies`, are valid while the plant is
referenced: a recorder holds its message, so a message holding its plant would form a loop
through Basilisk's C++ objects that the garbage collector leaves in place.

### `imu_spacecraft`: `ImuSpacecraft`

A rigid hub in a 500 km, 45 degree low Earth orbit under point-mass Earth gravity, tumbling, with
one IMU at its centre of mass. Its keyword defaults are the nominal vehicle:

| keyword | units, frame | nominal |
|---|---|---|
| `semi_major_axis_m` | m | 6871e3 |
| `eccentricity` | - | 0.0001 |
| `inclination_rad` | rad | 45 deg |
| `raan_rad` | rad | 48.2 deg |
| `arg_periapsis_rad` | rad | 347.8 deg |
| `true_anomaly_rad` | rad, at t = 0 | 85.3 deg |
| `mass_kg` | kg | 750 |
| `inertia_B_kg_m2` | kg m^2, about the centre of mass, body axes | diag(900, 800, 600) |
| `sigma0_BN` | MRP, body relative to inertial | (0.1, 0.2, -0.3) |
| `omega0_BN_B` | rad/s, body relative to inertial, body axes | 1 deg/s per axis |
| `gyro_bias_P` | rad/s, IMU platform frame | (1e-4, -2e-4, 5e-5) |
| `accel_bias_P` | m/s^2, IMU platform frame | (1e-4, -5e-5, 2e-4) |
| `gyro_noise_std_rad_s` | rad/s, per-sample 1-sigma per axis | 5e-5 |
| `accel_noise_std_m_s2` | m/s^2, per-sample 1-sigma per axis | 5e-5 |
| `step_s` | s, dynamics and sensor step | 1.0 |
| `noise_seed` | seed of the IMU's white-noise generator | 0 |

Standard dispersion, independent normal per axis around the default, drawn in this order:

| keyword | 1-sigma per axis |
|---|---|
| `sigma0_BN` | 0.01 |
| `omega0_BN_B` | 0.2 deg/s |
| `gyro_bias_P` | 1e-4 rad/s |
| `accel_bias_P` | 1e-4 m/s^2 |

`recorded_fields()`, dataset path to Basilisk message, field and units:

| dataset | Basilisk message | field | units |
|---|---|---|---|
| `truth/sigma_BN` | the hub's `SCStatesMsg` | `sigma_BN` | MRP |
| `truth/omega_BN_B` | the hub's `SCStatesMsg` | `omega_BN_B` | rad/s |
| `truth/r_N` | the hub's `SCStatesMsg` | `r_BN_N` | m |
| `truth/v_N` | the hub's `SCStatesMsg` | `v_BN_N` | m/s |
| `imu/gyro_P` | the IMU's `IMUSensorMsg` | `AngVelPlatform` | rad/s |
| `imu/accel_P` | the IMU's `IMUSensorMsg` | `AccelPlatform` | m/s^2 |

`readers()` holds role `imu`, an `IMUSensorMsgReader` on the IMU's `IMUSensorMsg`.
`initial_state()` returns `sigma0_BN` (MRP), `r0_N` (m), `v0_N` (m/s) and `omega0_BN_B` (rad/s).
The IMU sits at the hub's centre of mass with identity mounting, so its platform frame is the body
frame; Basilisk's `imuSensor` documentation describes the sensor model.

## Running one case

`run_case(plant, fsw_module, *, fields, duration_s, record_s=10.0)` adds one Basilisk module to
the plant, attaches a Basilisk recorder for each entry of `fields`, dataset path to
`(message, field, units)`, sampling every `record_s` from t = 0, runs, and returns the run file's
datasets: path to `(array, units)`, with `t_s` first. A flight-software module reads
`plant.readers()` and starts from `plant.initial_state()`; Basilisk's stock `inertial3D` guidance
module stands in for one here:

```python
>>> import tempfile
>>> from pathlib import Path
>>> from Basilisk.fswAlgorithms import inertial3D
>>> from wise_dynamics import write_run
>>> from wise_dynamics.basilisk import ImuSpacecraft, run_case
>>> plant = ImuSpacecraft()
>>> guidance = inertial3D.inertial3D()
>>> guidance.ModelTag = "guidance"
>>> fields = {
...     **plant.recorded_fields(),
...     "guidance/sigma_RN": (guidance.attRefOutMsg, "sigma_RN", "MRP"),
... }
>>> datasets = run_case(plant, guidance, fields=fields, duration_s=60.0)
>>> datasets["t_s"][0].tolist()
[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
>>> datasets["truth/r_N"][0].shape
(7, 3)
>>> scratch = tempfile.TemporaryDirectory()
>>> write_run(Path(scratch.name) / "run.h5", datasets, {"case": 0}).name
'run.h5'
>>> scratch.cleanup()

```

`tests/test_readme.py` runs this example.

## Run files

`write_run(path, datasets, attrs)` writes one HDF5 file: each dataset as float64 with a string
`units` attribute, the attributes at the root. It creates parent directories, writes a path once,
and leaves no file when a write fails. The plugin's `sim_run_writer` writes a case's
`<out_dir>/run_<NNNN>.h5` with these root attributes:

| attribute | value |
|---|---|
| `case` | the case number |
| `seed` | the session seed |
| `created_utc` | ISO 8601 UTC time of writing, to the second |
| `versions_bsk` | the installed Basilisk version |
| `versions_wise_dynamics` | the installed WiseDynamics version |
| `draw_<field>` | each draw, a float array |

`extra_attrs` adds further root attributes, such as the flight software's version as
`versions_<distribution>`, beside these; it may not reuse one of their names.

## pytest plugin

Installing `wise-dynamics` registers the plugin through its `pytest11` entry point, so a
consumer's `conftest.py` names nothing; naming the module in `pytest_plugins` as well makes
pytest raise "Plugin already registered under a different name".

| option | default | meaning |
|---|---|---|
| `--sim-runs` | 8 | cases in the session, at least 1 |
| `--sim-out` | a new directory under the system temporary directory | where run files go; a directory already holding run files is refused |
| `--sim-seed` | 0 | session seed |
| `--sim-duration-s` | 5700 | simulated seconds per case |
| `--sim-spacecraft` | `imu_spacecraft` | the plant's name in `PLANTS` |

The controlling process resolves the output directory to one absolute path and hands it to
every pytest-xdist worker; the first run file creates it, and the terminal summary names it.
`sim_run_writer` returns each run file's path, so a session-end check of the run files finds the
directory from them.

| fixture | gives |
|---|---|
| `sim_case_index` | the case number; a test using it, directly or through the per-case fixtures below, runs once per case |
| `sim_duration_s` | each case's simulated duration in seconds, `--sim-duration-s` |
| `sim_draws` | the case's draws; override it to supply a scenario's own |
| `sim_plant` | the case's plant, built with `sim_draws` as keywords and its sensor noise seeded from the case's stream |
| `sim_run_writer` | `writer(datasets, extra_attrs=None) -> Path`, writing the case's run file with `sim_draws` recorded |

Case `k` of seed `s` takes its random streams from
`numpy.random.SeedSequence(s, spawn_key=(k,)).spawn(2)`: the first draws its standard dispersion,
the second seeds its sensor noise. Case `k` is therefore the same whatever `--sim-runs` is, and a
recorded case reruns alone from its file's `seed` and `case` attributes:

```sh
pytest <scenario> --sim-seed S --sim-runs N "<scenario>::test_case[k]"
```

A scenario supplying its own draws overrides `sim_draws`; whichever draws a case has are both
what its plant is built with and what its run file records.

A consumer's scenario test:

```python
from wise_dynamics.basilisk import run_case


def test_case(sim_duration_s, sim_plant, sim_run_writer):
    fsw = build_flight_software(sim_plant.readers(), sim_plant.initial_state(), sim_plant.mu)
    fields = {**sim_plant.recorded_fields(), **estimate_fields(fsw)}
    datasets = run_case(sim_plant, fsw, fields=fields, duration_s=sim_duration_s)
    sim_run_writer(datasets, extra_attrs={"versions_flight_software": FSW_VERSION})
```

A Monte Carlo run is that test over many cases in one session: `pytest <scenario> -n 8
--sim-runs 852 --sim-out DIR`.

## Stable interface

A release changes these only as a breaking change: the names in the `__all__` of the two packages,
`wise_dynamics` and `wise_dynamics.basilisk`, imported from those packages; the `Plant` members in
[Plants](#plants); the keywords, reader role, initial-state keys and dataset names in the
`ImuSpacecraft` tables; the `run_case` signature in [Running one case](#running-one-case); the root
attributes in [Run files](#run-files); and the options, fixtures and case seeding in
[pytest plugin](#pytest-plugin). `tests/test_interface.py` and `tests/test_pytest_plugin.py` hold
them.
