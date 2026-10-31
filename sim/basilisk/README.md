# Basilisk

Basilisk is a locked dependency (`bsk[all]`) of this repository's uv project, installed from
prebuilt wheels. Basilisk is the first simulation host by ADR 0001 rev 2. This directory is a
plant library: a consumer that boots a named flight-software instance against one of these
plants pins a tagged release of this repository.

## Layout

- `models/` — reusable, bskSim-style dynamics stacks (spacecraft, gravity, sensors): the plant.
  Exposes its messages by role (`readers()`), its live initial condition (`initial_state()`),
  and `mu`, for a consumer's own host adapter to read; `default_draws(rng)` gives that model's
  standard per-case dispersion as its own constructor keyword arguments.
- `run.py` — `run_case(plant, fsw_module, fsw_outputs, duration_s, record_s=10.0)`: adds any
  Basilisk `SysModel` to `plant`'s task, records `plant.truth_messages()` plus the caller's own
  `fsw_outputs`, and runs. Returns the live recorders. Works with any plant that implements
  `sim`, `task_name`, `readers()` and `truth_messages()`.
- `runs.py` — `write_run(path, arrays, attrs)`: writes one HDF5 run file — root attrs plus a set
  of float64 datasets, each carrying a `units` attribute — from whatever dataset/attr names the
  caller supplies, creating `path`'s parent directory on first write if it does not exist yet.
- `pytest_plugin.py` — a pytest plugin registered via this package's own `pytest11` entry point:
  installing `wise-dynamics` is enough for pytest to load it. A consumer's own `pytest_plugins`
  must not also name `sim.basilisk.pytest_plugin` -- pytest refuses the same module registered
  under two names. `--sim-runs`/`--sim-out`/`--sim-seed`/`--sim-duration-s`/`--sim-spacecraft`
  options (namespaced so an autoloaded plugin cannot collide with a consumer's own option of the
  same short name), and `plant`/`draws`/`run_writer` fixtures (the named plant, built with its
  standard per-case draws, each case keyed off an independent child of the campaign seed; a
  callable recording the draws alongside the generic root attrs and writing `run_<NNNN>.h5` via
  `runs.py`). A consumer parametrizes its own test over `case_index` and builds its own
  flight-software module.

## Interface

The stable surface a consumer's own host adapter codes against, pinned by
`tests/test_interface.py`:

- `readers()` returns role `imu`.
- `truth_messages()` returns roles `truth` and `imu`.
- `initial_state()` returns exactly `sigma0_BN`, `r0_N`, `v0_N`, `omega0_BN_B`.
- `run_case(plant, fsw_module, fsw_outputs, duration_s, record_s=10.0)` (`run.py`).
- `write_run(path, arrays, attrs)` (`runs.py`).

## The IMU reader

`plant.readers()["imu"]` is an `IMUSensorMsgReader` on the plant's `imuSensor`, reporting the
non-gravitational specific force (confirmed empirically: zero bias/noise in orbit reads exactly
zero against real gravitational acceleration) in its own platform frame — the body frame only
when the sensor's mounting is identity, as this plant's default is. Its first sample, at t=0 s,
reads zero: Basilisk writes a sensor's first output after the first dynamics step, not at
`Reset()`.
