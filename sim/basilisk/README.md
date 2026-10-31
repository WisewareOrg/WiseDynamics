# Basilisk

Basilisk is a locked dependency (`bsk[all]`) of this repository's uv project, installed from
prebuilt wheels. Basilisk is the first simulation host by ADR 0001 rev 2. This directory depends
on no flight software and hosts no algorithm; a consumer that boots a named flight-software
instance against one of these plants pins a tagged release of this repository.

## Layout

- `models/` — reusable, bskSim-style dynamics stacks (spacecraft, gravity, sensors): the plant.
  Exposes its messages by role (`readers()`), its live initial condition (`initial_state()`),
  and `mu`, for a consumer's own host adapter to read; `default_draws(rng)` gives that model's
  standard per-case dispersion as its own constructor keyword arguments.
- `run.py` — `run_case(plant, fsw_module, duration_s, record_s=10.0)`: adds any Basilisk
  `SysModel` to `plant`'s task, records the plant's truth and IMU messages plus the module's own
  output messages by role, and runs. Returns the live recorders.
- `runs.py` — `write_run(path, arrays, attrs)`: writes one HDF5 run file per SPEC-demo.md §2's
  layout (root attrs; float64 datasets, each carrying a `units` attribute), from whatever
  dataset/attr names the caller supplies. Carries no knowledge of which names a particular study
  needs.
- `pytest_plugin.py` — a pytest plugin a consumer loads with `pytest_plugins =
  ["sim.basilisk.pytest_plugin"]`: `--runs`/`--out`/`--seed`/`--duration-s`/`--spacecraft`
  options, and `plant`/`draws`/`run_writer` fixtures (the named plant, built with its standard
  per-case draws; a callable pre-filling the generic root attrs and writing `run_<NNNN>.h5` via
  `runs.py`). Carries no algorithm knowledge: a consumer parametrizes its own test over
  `case_index` and builds its own flight-software module.

## The IMU reader

`plant.readers()["imu"]` is an `IMUSensorMsgReader` on the plant's `imuSensor`, reporting the
non-gravitational specific force (confirmed empirically: zero bias/noise in orbit reads exactly
zero against real gravitational acceleration) in its own platform frame — the body frame only
when the sensor's mounting is identity, as this plant's default is. Its first sample, at t=0 s,
reads zero: Basilisk writes a sensor's first output after the first dynamics step, not at
`Reset()`.
