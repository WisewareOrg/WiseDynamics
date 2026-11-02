# wise_dynamics.basilisk

Basilisk spacecraft plants, the one-case runner, and the campaign machinery and pytest plugin
that drive them. Basilisk is the first simulation host by ADR 0001 rev 1. This file is the
reference for the `wise-dynamics` distribution; each symbol's docstring carries its full
contract.

## Modules

| module | holds |
|---|---|
| `wise_dynamics/campaign.py` | `Campaign` and `Case`: seeding, draws, the plant a case builds, its run file; `distribution_version` |
| `wise_dynamics/dispersion.py` | `Dispersion` and `apply_draws`: per-case normal dispersion of a configuration's 3-vector fields |
| `wise_dynamics/run_file.py` | `Dataset` and `write_run`: one HDF5 run file |
| `wise_dynamics/pytest_plugin.py` | the pytest plugin: `--sim-*` options, `sim_*` fixtures, `campaign_from_config` |
| `wise_dynamics/basilisk/plant.py` | `Plant`, the contract every plant meets; `RecordedField` |
| `wise_dynamics/basilisk/runner.py` | `run_case` and `Recording` |
| `wise_dynamics/basilisk/plants/` | `PLANTS`, the registry by name, and one module per plant |

`wise_dynamics` re-exports `Campaign`, `Case`, `Dataset`, `Dispersion`, `apply_draws`,
`distribution_version` and `write_run`. `wise_dynamics.basilisk` re-exports `Plant`,
`RecordedField`, `Recording`, `run_case`, `PLANTS` and the `ImuSpacecraft` classes.

## Plants

A plant is a `Plant` subclass: it owns its own Basilisk simulation with one process and one
task, built in its constructor `(config=None, *, noise_seed=0)` from a frozen configuration
dataclass. Each plant class provides:

- `nominal()`, the vehicle's nominal configuration;
- `standard_dispersion`, the vehicle's standard per-case dispersion;
- `recorded_fields`, run-file dataset name to recorded message field and units;
- `mu`, the central body's gravitational parameter in m^3/s^2;
- `readers()`, sensor outputs by role as subscribed Basilisk message readers;
- `truth_messages()`, recordable output messages by role;
- `initial_state()`, the initial condition by key;
- `add_model(module)` and `execute(duration_s)`. A plant runs once.

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
>>> from wise_dynamics import Campaign
>>> from wise_dynamics.basilisk import RecordedField, run_case
>>> campaign = Campaign(
...     "imu_spacecraft", seed=0, runs=852, duration_s=60.0,
...     out_dir=Path(tempfile.mkdtemp()) / "campaign",
... )
>>> case = campaign.case(417)
>>> plant = case.build_plant()
>>> guidance = inertial3D.inertial3D()
>>> guidance.ModelTag = "guidance"
>>> recording = run_case(
...     plant, guidance, fsw_outputs={"guidance": guidance.attRefOutMsg},
...     duration_s=campaign.duration_s,
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
>>> case.write(datasets).name
'run_0417.h5'

```

`tests/test_readme.py` runs this example.

## Campaigns

`Campaign(plant, seed, runs, duration_s, out_dir)` is a set of cases of the plant named in
`PLANTS`. Case `k` takes its random streams from
`numpy.random.SeedSequence(seed, spawn_key=(k,)).spawn(2)`: the first draws the case's dispersed
configuration, the second seeds its sensor noise. Case `k` is therefore the same whatever `runs`
is, and `campaign.case(k)` rebuilds any recorded case from its file's `seed` and `case`
attributes.

`campaign.case(k)` applies the standard dispersion; `campaign.case(k, draws)` applies a
scenario's own, a mapping of configuration field name to value, drawn for example from
`campaign.draws_rng(k)`. Either way the case's `draws` are both what its plant is built with and
what its run file records.

## Run files

`write_run(path, datasets, attrs)` writes one HDF5 file: each dataset as float64 with a string
`units` attribute, the attributes at the root. It creates parent directories and writes a path
once. `Case.write(datasets, extra_attrs=None)` writes `<out_dir>/run_<NNNN>.h5` with these root
attributes:

| attribute | value |
|---|---|
| `case` | the case number |
| `seed` | the campaign seed |
| `created_utc` | ISO 8601 UTC time of writing, to the second |
| `producer` | `wise_dynamics.basilisk` |
| `versions_bsk` | the installed Basilisk version |
| `versions_wise_dynamics` | the installed version, with `+<commit>` for an install from a git source |
| `draw_<field>` | each draw, a float array |

`extra_attrs` adds further root attributes, such as the flight software's version as
`versions_<distribution>`, beside these.

## pytest plugin

Installing `wise-dynamics` registers the plugin through its `pytest11` entry point, so a
consumer's `conftest.py` names nothing; naming the module in `pytest_plugins` as well makes
pytest raise "Plugin already registered under a different name". The plugin imports Basilisk
only when a `sim_` fixture builds a plant.

| option | default | meaning |
|---|---|---|
| `--sim-runs` | 8 | cases in the campaign, at least 1 |
| `--sim-out` | a new directory under the system temporary directory | where run files go; a directory already holding run files is refused |
| `--sim-seed` | 0 | campaign seed |
| `--sim-duration-s` | 5700 | simulated seconds per case |
| `--sim-spacecraft` | `imu_spacecraft` | the plant's name in `PLANTS` |

The controlling process resolves the output directory to one absolute path and hands it to
every pytest-xdist worker; the first run file creates it, and the terminal summary names it.

| fixture | gives |
|---|---|
| `sim_case_index` | the case number; a test using it, directly or through the fixtures below, runs once per case |
| `sim_campaign` | the session's `Campaign` (session scope) |
| `sim_draws` | the case's draws; override it to supply a scenario's own |
| `sim_plant` | the case's plant, built from `sim_draws` |
| `sim_run_writer` | `writer(datasets, extra_attrs=None) -> Path`, the case's `Case.write` |

`campaign_from_config(config)` returns the session's `Campaign` from any hook, for example a
`pytest_sessionfinish` that checks the campaign directory. A consumer's scenario test:

```python
from wise_dynamics.basilisk import RecordedField, run_case


def test_case(sim_campaign, sim_plant, sim_run_writer):
    fsw = build_flight_software(sim_plant.readers(), sim_plant.initial_state(), sim_plant.mu)
    recording = run_case(
        sim_plant, fsw, fsw_outputs=fsw.outputs, duration_s=sim_campaign.duration_s
    )
    datasets = recording.datasets({**sim_plant.recorded_fields, **ESTIMATE_FIELDS})
    sim_run_writer(datasets, extra_attrs={"versions_flight_software": FSW_VERSION})
```

A campaign is that test at scale: `pytest <scenario> -n 8 --sim-runs 852 --sim-out DIR`.

## Stable interface

A release changes these only as a breaking change: the import paths and names in
[Modules](#modules); the `Plant` members in [Plants](#plants); the roles, initial-state keys and
dataset names in the `ImuSpacecraft` tables; the `run_case` and `Recording` signatures in
[Running one case](#running-one-case); the case seeding in [Campaigns](#campaigns); the root
attributes in [Run files](#run-files); and the options, fixtures and `campaign_from_config` in
[pytest plugin](#pytest-plugin). `tests/test_interface.py` and `tests/test_pytest_plugin.py`
hold them.
