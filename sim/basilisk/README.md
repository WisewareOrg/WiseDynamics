# Basilisk

Basilisk is a locked dependency (`bsk[all]`) of this repository's uv project, installed from prebuilt
wheels. Scenarios live here, importing GNC algorithms
from [WiseGncDesign](https://github.com/WisewareOrg/WiseGncDesign) rather than reimplementing
them; nothing algorithmic lives under this directory. Basilisk is the first simulation host by
ADR 0001 rev 2. `gnc-algorithms` is an editable path dependency on a sibling `WiseGncDesign`
checkout, built fresh from whatever revision sits beside this repository.

## Layout

- `models/` — reusable, bskSim-style dynamics stacks (spacecraft, gravity, sensors); the plant,
  carrying no algorithm.
- `hosting/` — `AlgorithmModule`, one generic Basilisk `SysModel` that hosts any WiseGncDesign
  algorithm through declarative field maps; the seam, carrying no vocabulary belonging to any
  one algorithm, sensor, or dynamics model, enforced by `scripts/check-hosting-generic.sh`.
- `scenarios/` — one `test_<scenario>.py` module per study, each a parametrized pytest module
  that wires a `models/` plant to an `AlgorithmModule` binding (declared in `bindings.py`), owns
  its own dispersions, and writes one `run_<NNNN>.h5` per case straight into `--out`. A single
  nominal case is `execute_scenario(create_scenario())`, a plain function call; a campaign is
  this directory run as pytest, `pytest sim/basilisk/scenarios --runs N --out DIR [-n WORKERS]`.
- `conftest.py` — the campaign's `--runs`/`--out`/`--seed`/`--duration-s` options and the
  `plant`/`algorithm_binding`/`draws` fixtures every study's cases are built from, plus the
  session-end check of every run against WiseGncAnalysis's own loader.

## Exported sensor datasets

`imu/gyro_P`/`imu/accel_P` in a run file carry the IMU's reading in its own platform frame,
which equals the body frame only when the sensor's mounting is identity, as in this demo. Their
first sample, at t=0 s, reads zero: Basilisk writes a sensor's first output after the first
dynamics step, not at `Reset()`.
