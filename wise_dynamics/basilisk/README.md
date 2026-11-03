# wise_dynamics.basilisk

Basilisk spacecraft vehicles, the one-case runner, the run file, and the pytest plugin whose
session over `--sim-runs` cases is a Monte Carlo run. Basilisk is the first simulation host by
ADR 0001 rev 1. Each symbol's docstring carries its arguments, returns and raises.

## Modules

| module | holds |
|---|---|
| `wise_dynamics/run_file.py` | `RunFile`: one HDF5 run file, written once, float64 datasets with a `units` attribute |
| `wise_dynamics/pytest_plugin.py` | the pytest plugin's options and `sim_*` fixtures |
| `wise_dynamics/basilisk/vehicle.py` | `Vehicle`: the interface every vehicle meets |
| `wise_dynamics/basilisk/runner.py` | `run_case`: runs one vehicle with one flight-software module |
| `wise_dynamics/basilisk/vehicles/__init__.py` | `VEHICLES`: the registry of vehicles by name |
| `wise_dynamics/basilisk/vehicles/imu_spacecraft.py` | `ImuSpacecraft`: the IMU spacecraft |

Consumers import from `wise_dynamics` and `wise_dynamics.basilisk`.

## `imu_spacecraft`: `ImuSpacecraft`

A rigid hub in a 500 km, 45 degree low Earth orbit under Basilisk's point-mass Earth gravity,
`ImuSpacecraft.mu` = 3.986004360e14 m^3/s^2, tumbling, with one IMU at its centre of mass and
identity mounting, so its platform frame is the body frame. Each keyword defaults to the class
attribute of the same name, so `ImuSpacecraft.step_s` reads the nominal step without building a
vehicle:

| keyword | units, frame | nominal | standard dispersion, 1-sigma per axis |
|---|---|---|---|
| `semi_major_axis_m` | m | 6871e3 | |
| `eccentricity` | - | 0.0001 | |
| `inclination_rad` | rad | 45 deg | |
| `raan_rad` | rad | 48.2 deg | |
| `arg_periapsis_rad` | rad | 347.8 deg | |
| `true_anomaly_rad` | rad, at t = 0 | 85.3 deg | |
| `mass_kg` | kg | 750 | |
| `inertia_B_kg_m2` | kg m^2, about the centre of mass, body axes | diag(900, 800, 600) | |
| `sigma0_BN` | MRP, body relative to inertial | (0.1, 0.2, -0.3) | 0.01 |
| `omega0_BN_B_rad_s` | rad/s, body relative to inertial, body axes | 1 deg/s per axis | 0.2 deg/s |
| `gyro_bias_P_rad_s` | rad/s, IMU platform frame | (1e-4, -2e-4, 5e-5) | 1e-4 rad/s |
| `accel_bias_P_m_s2` | m/s^2, IMU platform frame | (1e-4, -5e-5, 2e-4) | 1e-4 m/s^2 |
| `gyro_noise_std_rad_s` | rad/s, per-sample 1-sigma per axis | 5e-5 | |
| `accel_noise_std_m_s2` | m/s^2, per-sample 1-sigma per axis | 5e-5 | |
| `step_s` | s, dynamics and sensor step | 1.0 | |
| `noise_seed` | seed of the IMU's white-noise generator | 0 | |

`readers()` holds role `imu`, an `IMUSensorMsgReader`. `initial_state()` returns `sigma0_BN`
(MRP), `r0_N` (m), `v0_N` (m/s) and `omega0_BN_B` (rad/s). `recorded_fields()` maps each dataset
to its `(message, field, units)`: `truth/sigma_BN` (MRP), `truth/omega_BN_B` (rad/s), `truth/r_N`
(m), `truth/v_N` (m/s), `imu/gyro_P` (rad/s) and `imu/accel_P` (m/s^2).

## pytest plugin

Installing `wise-dynamics` registers the plugin through its `pytest11` entry point, so a
consumer's `conftest.py` names nothing.

| option | default | meaning |
|---|---|---|
| `--sim-runs` | 8 | cases in the session, at least 1 |
| `--sim-out` | a new directory under the system temporary directory | where run files go; a directory already holding run files is refused |
| `--sim-seed` | 0 | session seed |
| `--sim-duration-s` | 5700 | simulated seconds per case |
| `--sim-vehicle` | `imu_spacecraft` | the vehicle's name in `VEHICLES` |

| fixture | gives |
|---|---|
| `sim_case_index` | the case number; a test using it, directly or through the fixtures below, runs once per case |
| `sim_duration_s` | `--sim-duration-s` |
| `sim_draws` | the case's draws, by default its standard dispersion; override it to supply a scenario's own |
| `sim_vehicle` | the case's vehicle, built with `sim_draws` as keywords and its noise seeded from the case |
| `sim_run_file` | the case's `RunFile`, `run_<NNNN>.h5` under `--sim-out`, whose `write(datasets, attrs=None)` adds `attrs` beside its root attributes `case`, `seed`, `created_utc`, `versions_bsk`, `versions_wise_dynamics` and `draw_<keyword>` per draw |

Case `k` of seed `s` takes its draws and its noise seed from the two streams of
`numpy.random.SeedSequence(s, spawn_key=(k,)).spawn(2)`, so it is the same whatever `--sim-runs`
is and reruns alone: `pytest <scenario> --sim-seed S --sim-runs N "<scenario>::test_case[k]"`.

A scenario test, with Basilisk's stock `inertial3D` guidance standing in for flight software that
reads `sim_vehicle.readers()` and starts from `sim_vehicle.initial_state()`; `pytest <scenario>
-n 8 --sim-runs 852 --sim-out DIR` runs it as a Monte Carlo run:

```python
from Basilisk.fswAlgorithms import inertial3D

from wise_dynamics.basilisk import ImuSpacecraft, run_case

RECORD_S = 10 * ImuSpacecraft.step_s


def test_case(sim_vehicle, sim_duration_s, sim_run_file):
    guidance = inertial3D.inertial3D()
    guidance.ModelTag = "guidance"
    fields = {
        **sim_vehicle.recorded_fields(),
        "guidance/sigma_RN": (guidance.attRefOutMsg, "sigma_RN", "MRP"),
    }
    datasets = run_case(
        sim_vehicle, guidance, fields=fields, duration_s=sim_duration_s, record_s=RECORD_S
    )
    sim_run_file.write(datasets, {"versions_guidance": "1.0"})
```
