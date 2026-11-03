# wise_dynamics.basilisk

Basilisk vehicles, the scenarios that place them, and the pytest plugin whose session over
`--sim-runs` cases is a Monte Carlo run. A vehicle is mass properties, sensors and logging, and
knows nothing about where it flies. A scenario is a prebuilt operating environment and the
placement of the vehicles added to it, and runs them once with a flight-software module. Basilisk is the first
simulation host by ADR 0001 rev 1. Each symbol's docstring carries its arguments, returns and
raises.

## Modules

| module | holds |
|---|---|
| `wise_dynamics/pytest_plugin.py` | the pytest plugin's options and `sim_*` fixtures |
| `wise_dynamics/basilisk/vehicle.py` | `Vehicle`: the interface every vehicle meets, its draws and its run file |
| `wise_dynamics/basilisk/scenario.py` | `Scenario`: the interface every scenario meets, and its one run |
| `wise_dynamics/basilisk/vehicles/__init__.py` | `VEHICLES`: the vehicles by name |
| `wise_dynamics/basilisk/vehicles/imu_spacecraft.py` | `ImuSpacecraft`: the IMU spacecraft |
| `wise_dynamics/basilisk/scenarios/__init__.py` | `SCENARIOS`: the scenarios by name |
| `wise_dynamics/basilisk/scenarios/leo500.py` | `Leo500`: point-mass Earth, 500 km, 45 degrees |

Consumers import from `wise_dynamics.basilisk`. Each keyword of a vehicle defaults to the class
attribute of the same name, and a scenario is its class attributes, so `ImuSpacecraft.mass_kg` and
`Leo500.step_s` read the nominal values without building anything.

## Vehicle `imu_spacecraft`: `ImuSpacecraft(**keywords)`

A rigid hub with one IMU at its centre of mass and identity mounting, so its platform frame is the
body frame.

| keyword | units, frame | nominal | standard dispersion, 1-sigma per axis |
|---|---|---|---|
| `mass_kg` | kg | 750 | |
| `inertia_B_kg_m2` | kg m^2, about the centre of mass, body axes | diag(900, 800, 600) | |
| `gyro_bias_P_rad_s` | rad/s, IMU platform frame | (1e-4, -2e-4, 5e-5) | 1e-4 rad/s |
| `accel_bias_P_m_s2` | m/s^2, IMU platform frame | (1e-4, -5e-5, 2e-4) | 1e-4 m/s^2 |
| `gyro_noise_std_rad_s` | rad/s, per-sample 1-sigma per axis | 5e-5 | |
| `accel_noise_std_m_s2` | m/s^2, per-sample 1-sigma per axis | 5e-5 | |
| `noise_seed` | seed of the IMU's white-noise generator | 0 | |

`ImuSpacecraft.draw(rng)` returns the standard dispersion's keywords, in table order, as nominal
plus normal samples from the numpy `Generator`. `readers()` holds role `imu`, an
`IMUSensorMsgReader`. `recorded_fields()` maps each dataset to its `(message, field, units)`:
`truth/sigma_BN` (MRP), `truth/omega_BN_B` (rad/s), `truth/r_N` (m, in N), `truth/v_N` (m/s, in
N), `imu/gyro_P` (rad/s) and `imu/accel_P` (m/s^2). `truth/r_N` and `truth/v_N` are the
body-frame origin's state, which is the centre of mass's state because the hub's centre of mass
sits at that origin. `write_run_file(path, datasets, attrs=None)` writes one HDF5 file, each
dataset float64 with a `units` attribute.

## Scenario `leo500`: `Leo500()`

Basilisk's point-mass Earth, `Leo500.mu_m3_s2` = `{"earth": 3.986004360e14}`, with the vehicle
tumbling on a classical-element orbit. The inertial frame N is Earth-centred and non-rotating,
with no epoch, since point-mass gravity has nothing that depends on time. The constructor takes
no arguments; these class attributes define the scenario:

| attribute | units, frame | value | standard dispersion, 1-sigma per axis |
|---|---|---|---|
| `semi_major_axis_m` | m | 6871e3 | |
| `eccentricity` | - | 0.0001 | |
| `inclination_rad` | rad | 45 deg | |
| `raan_rad` | rad | 48.2 deg | |
| `arg_periapsis_rad` | rad | 347.8 deg | |
| `true_anomaly_rad` | rad, at t = 0 | 85.3 deg | |
| `sigma0_BN` | MRP, body relative to N | (0.1, 0.2, -0.3) | 0.01 |
| `omega0_BN_B_rad_s` | rad/s, body relative to N, body axes | 1 deg/s per axis | 0.2 deg/s |
| `step_s` | s, integration and sensor step | 1.0 | |

`Leo500.draw(rng)` draws its standard dispersion as a vehicle's `draw` does, and
`disperse(draws)` sets those values on one instance; any other attribute is refused.
`add(vehicle)` places a vehicle at the scenario's placement; a second vehicle is refused until it
can be given a placement of its own. `initial_state()` returns the vehicle's `sigma0_BN` (MRP),
`r0_N` (m, in N), `v0_N` (m/s, in N) and `omega0_BN_B` (rad/s), valid before the run.
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
