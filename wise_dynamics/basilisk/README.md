# wise_dynamics.basilisk

Basilisk spacecraft plants, the one-case runner, the run-file writer, and the pytest plugin
whose session over `--sim-runs` cases is a Monte Carlo run. Basilisk is the first
simulation host by ADR 0001 rev 1. This file is the reference for the `wise-dynamics`
distribution; each symbol's docstring carries its full contract.

## Modules

Each module holds one class, or the functions of one concern, and is named for what it holds.

| module | holds |
|---|---|
| `wise_dynamics/dataclass_config.py` | `DataclassConfig`: the shape of any dataclass instance |
| `wise_dynamics/dispersion.py` | `Dispersion` and its inverse `apply_draws`: per-case normal dispersion of a configuration's 3-vector fields |
| `wise_dynamics/dataset.py` | `Dataset`: one array with its units |
| `wise_dynamics/run_file.py` | `write_run`: one HDF5 run file |
| `wise_dynamics/provenance.py` | `distribution_version`: the installed version a run file records |
| `wise_dynamics/session_options.py` | `SessionOptions`: a pytest session's parsed `--sim-*` options |
| `wise_dynamics/run_writer.py` | `RunWriter`: the type of the `sim_run_writer` fixture's value |
| `wise_dynamics/pytest_plugin.py` | the pytest plugin's hooks and `sim_*` fixtures |
| `wise_dynamics/basilisk/plant.py` | `Plant`: the contract every plant meets |
| `wise_dynamics/basilisk/plant_config.py` | `PlantConfig`: the shape of a plant's configuration |
| `wise_dynamics/basilisk/recorded_field.py` | `RecordedField`: one recorded message field with its units |
| `wise_dynamics/basilisk/recording.py` | `Recording`: one case's recorders and the datasets they hold |
| `wise_dynamics/basilisk/runner.py` | `run_case`: runs one case |
| `wise_dynamics/basilisk/orbit_elements.py` | `OrbitElements`: an initial orbit as classical elements |
| `wise_dynamics/basilisk/rigid_hub.py` | `RigidHub`: a hub's mass and inertia |
| `wise_dynamics/basilisk/vectors.py` | `Vector3`, `Matrix3` and `column()`: configuration vectors and Basilisk's column form |
| `wise_dynamics/basilisk/swig.py` | type aliases for Basilisk's SWIG classes |
| `wise_dynamics/basilisk/plants/` | `PLANTS`, the registry by name, and one package per plant |
| `wise_dynamics/basilisk/plants/imu_spacecraft/` | `config.py`: `ImuSpacecraftConfig`; `plant.py`: `ImuSpacecraft` |

Consumers import from the two packages, `wise_dynamics` and `wise_dynamics.basilisk`; each
declares its public names in `__all__`, and those import paths stay the same whichever module a
name lives in.

## Plants

A plant is a `Plant` subclass: it owns its own Basilisk simulation with one process and one
task, built by the constructor `(config=None, *, noise_seed=0)` from a frozen configuration
dataclass, which carries the step as `step_s`. A subclass adds its models in `_build(config,
noise_seed)`. Each plant class provides:

- `nominal()`, the vehicle's nominal configuration;
- `standard_dispersion`, the vehicle's standard per-case dispersion;
- `recorded_fields`, run-file dataset name to recorded message field and units;
- `mu`, a class attribute: the central body's gravitational parameter its gravity model uses,
  in m^3/s^2;
- `readers()`, sensor outputs by role as subscribed Basilisk message readers;
- `truth_messages()`, recordable output messages by role;
- `initial_state()`, the initial condition by key;
- `config`, the configuration it was built from;
- `add_model(module)` and `execute(duration_s)`. A plant runs once.

Each reader `readers()` returns holds its plant, so it reads the plant's live output for as long
as it is held. The messages `truth_messages()` returns, `spacecraft` and `imu`, along with
anything reached through them such as `spacecraft.gravField.gravBodies`, are valid while the
plant is referenced: a recorder holds its message, so a message holding its plant would form a
loop through Basilisk's C++ objects that the garbage collector leaves in place. A `Recording`
holds its own data and outlives its plant.

### `imu_spacecraft`: `ImuSpacecraft`

A rigid hub in a 500 km, 45 degree low Earth orbit under point-mass Earth gravity, tumbling, with
one IMU at its centre of mass. `ImuSpacecraftConfig()` is the nominal vehicle:

| field | units, frame | nominal |
|---|---|---|
| `orbit.semi_major_axis_m` | m | 6871e3 |
| `orbit.eccentricity` | - | 0.0001 |
| `orbit.inclination_rad` | rad | 45 deg |
| `orbit.raan_rad` | rad | 48.2 deg |
| `orbit.arg_periapsis_rad` | rad | 347.8 deg |
| `orbit.true_anomaly_rad` | rad, at t = 0 | 85.3 deg |
| `hub.mass_kg` | kg | 750 |
| `hub.inertia_B_kg_m2` | kg m^2, about the centre of mass, body axes | diag(900, 800, 600) |
| `sigma0_BN` | MRP, body relative to inertial | (0.1, 0.2, -0.3) |
| `omega0_BN_B` | rad/s, body relative to inertial, body axes | 1 deg/s per axis |
| `gyro_bias_P` | rad/s, IMU platform frame | (1e-4, -2e-4, 5e-5) |
| `accel_bias_P` | m/s^2, IMU platform frame | (1e-4, -5e-5, 2e-4) |
| `gyro_noise_std_rad_s` | rad/s, per-sample 1-sigma per axis | 5e-5 |
| `accel_noise_std_m_s2` | m/s^2, per-sample 1-sigma per axis | 5e-5 |
| `step_s` | s, dynamics and sensor step | 1.0 |

Standard dispersion, independent normal per axis around nominal, drawn in this order:

| field | 1-sigma per axis |
|---|---|
| `sigma0_BN` | 0.01 |
| `omega0_BN_B` | 0.2 deg/s |
| `gyro_bias_P` | 1e-4 rad/s |
| `accel_bias_P` | 1e-4 m/s^2 |

Messages and their recorded fields:

| role | Basilisk message | reader in `readers()` | recorder field | dataset | units |
|---|---|---|---|---|---|
| `truth` | `SCStatesMsg` | | `sigma_BN` | `truth/sigma_BN` | MRP |
| `truth` | | | `omega_BN_B` | `truth/omega_BN_B` | rad/s |
| `truth` | | | `r_BN_N` | `truth/r_N` | m |
| `truth` | | | `v_BN_N` | `truth/v_N` | m/s |
| `imu` | `IMUSensorMsg` | `IMUSensorMsgReader` | `AngVelPlatform` | `imu/gyro_P` | rad/s |
| `imu` | | | `AccelPlatform` | `imu/accel_P` | m/s^2 |

`initial_state()` returns `sigma0_BN` (MRP), `r0_N` (m), `v0_N` (m/s) and `omega0_BN_B` (rad/s).
The IMU is mounted with identity `dcm_PB`, so its platform frame is the body frame. The
accelerometer reports non-gravitational specific force, which in free fall is its bias plus
noise. Both instruments read zero at t = 0, since Basilisk writes the IMU's first output after
the first step.

## Running one case

`run_case(plant, fsw_module, *, fsw_outputs, duration_s, record_s=10.0)` adds one Basilisk
module to the plant, records the plant's truth messages and the module's `fsw_outputs` every
`record_s` from t = 0, runs, and returns a `Recording`. Indexing a `Recording` by role gives the
Basilisk recorder; `datasets(fields)` gives the unit-tagged datasets of a run file, with `t_s`
first. A flight-software module reads `plant.readers()` and starts from
`plant.initial_state()`; Basilisk's stock `inertial3D` guidance module stands in for one here:

```python
>>> import tempfile
>>> from pathlib import Path
>>> from Basilisk.fswAlgorithms import inertial3D
>>> from wise_dynamics import write_run
>>> from wise_dynamics.basilisk import ImuSpacecraft, RecordedField, run_case
>>> plant = ImuSpacecraft()
>>> guidance = inertial3D.inertial3D()
>>> guidance.ModelTag = "guidance"
>>> recording = run_case(
...     plant, guidance, fsw_outputs={"guidance": guidance.attRefOutMsg}, duration_s=60.0
... )
>>> fields = {
...     **plant.recorded_fields,
...     "guidance/sigma_RN": RecordedField("guidance", "sigma_RN", "MRP"),
... }
>>> datasets = recording.datasets(fields)
>>> datasets["t_s"].data.tolist()
[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
>>> datasets["truth/r_N"].data.shape
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
| `producer` | `wise_dynamics.basilisk` |
| `versions_bsk` | the installed Basilisk version |
| `versions_wise_dynamics` | the installed version, with `+<commit>` for an install from a git source |
| `draw_<field>` | each draw, a float array |

`extra_attrs` adds further root attributes, such as the flight software's version as
`versions_<distribution>`, beside these; it may not reuse one of their names.

## pytest plugin

Installing `wise-dynamics` registers the plugin through its `pytest11` entry point, so a
consumer's `conftest.py` names nothing; naming the module in `pytest_plugins` as well makes
pytest raise "Plugin already registered under a different name". The plugin imports Basilisk
when pytest collects a test that uses `sim_case_index`, directly or through another per-case
`sim_` fixture.

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
| `sim_draws_rng` | a fresh numpy `Generator` on the case's draws stream |
| `sim_draws` | the case's draws; override it to supply a scenario's own |
| `sim_plant` | the case's plant: the nominal configuration with `sim_draws` applied, its sensor noise seeded from the case's stream |
| `sim_run_writer` | `writer(datasets, extra_attrs=None) -> Path`, writing the case's run file with `sim_draws` recorded |

Case `k` of seed `s` takes its random streams from
`numpy.random.SeedSequence(s, spawn_key=(k,)).spawn(2)`: the first draws its standard dispersion,
the second seeds its sensor noise. Case `k` is therefore the same whatever `--sim-runs` is, and a
recorded case reruns alone from its file's `seed` and `case` attributes:

```sh
pytest <scenario> --sim-seed S --sim-runs N "<scenario>::test_case[k]"
```

A scenario supplying its own draws overrides `sim_draws` and draws them from `sim_draws_rng`,
the stream the standard draws come from; whichever draws a case has are both what its plant is
built with and what its run file records:

```python
import pytest


@pytest.fixture
def sim_draws(sim_draws_rng):
    return {"gyro_bias_P": sim_draws_rng.normal(0.0, 1e-3, size=3)}
```

A consumer's scenario test:

```python
from wise_dynamics.basilisk import run_case


def test_case(sim_duration_s, sim_plant, sim_run_writer):
    fsw = build_flight_software(sim_plant.readers(), sim_plant.initial_state(), sim_plant.mu)
    recording = run_case(
        sim_plant,
        fsw,
        fsw_outputs=fsw.outputs,
        duration_s=sim_duration_s,
    )
    datasets = recording.datasets({**sim_plant.recorded_fields, **ESTIMATE_FIELDS})
    sim_run_writer(datasets, extra_attrs={"versions_flight_software": FSW_VERSION})
```

A Monte Carlo run is that test over many cases in one session: `pytest <scenario> -n 8
--sim-runs 852 --sim-out DIR`.

## Stable interface

A release changes these only as a breaking change: the import paths and names in
[Modules](#modules); the `Plant` members in [Plants](#plants); the roles, initial-state keys and
dataset names in the `ImuSpacecraft` tables; the `run_case` and `Recording` signatures in
[Running one case](#running-one-case); the root attributes in [Run files](#run-files); and the
options, fixtures and case seeding in [pytest plugin](#pytest-plugin).
`tests/test_interface.py` and `tests/test_pytest_plugin.py` hold them.
