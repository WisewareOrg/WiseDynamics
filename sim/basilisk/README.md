# Basilisk

Basilisk is a locked dependency (`bsk[all]`) of this repository's uv project, installed from
prebuilt wheels. Basilisk is the first simulation host by ADR 0001 rev 2. A study here runs a
spacecraft configuration with a named flight-software instance, booted through WiseGncDesign's
C&DH partition (`cdh.basilisk.make_fsw`); nothing algorithmic, and no host adapter, lives under
this directory. `gnc-algorithms[basilisk]` is an editable path dependency on a sibling
`WiseGncDesign` checkout, built fresh from whatever revision sits beside this repository.

## Layout

- `models/` — reusable, bskSim-style dynamics stacks (spacecraft, gravity, sensors): the plant.
  Exposes its messages by role (`readers()`) and its live initial condition (`initial_state()`)
  for a C&DH host adapter to read, and carries no algorithm.
- `scenarios/` — one `test_<scenario>.py` module per study, each a parametrized pytest module
  that builds a `models/` plant, boots a named flight-software instance onto it via
  `cdh.basilisk.make_fsw`, owns its own dispersions, and writes one `run_<NNNN>.h5` per case
  straight into `--out`. A single nominal case is `execute_scenario(create_scenario())`, a plain
  function call; a campaign is this directory run as pytest, `pytest sim/basilisk/scenarios
  --runs N --out DIR [-n WORKERS]`.
- `conftest.py` — the campaign's `--runs`/`--out`/`--seed`/`--duration-s`/`--spacecraft`/`--fsw`
  options and the `plant`/`fsw_name`/`draws` fixtures every study's cases are built from, plus
  the session-end check of every run against WiseGncAnalysis's own loader.

## Exported sensor datasets

`imu/gyro_P`/`imu/accel_P` in a run file carry the IMU's reading in its own platform frame,
which equals the body frame only when the sensor's mounting is identity, as in this demo. Their
first sample, at t=0 s, reads zero: Basilisk writes a sensor's first output after the first
dynamics step, not at `Reset()`.
