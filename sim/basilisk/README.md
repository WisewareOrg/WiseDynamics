# Basilisk

Basilisk is a locked dependency (`bsk[all]`) of this repository's uv project, installed from prebuilt
wheels. Scenarios live here, importing GNC algorithms
from [WiseGncDesign](https://github.com/WisewareOrg/WiseGncDesign) rather than reimplementing
them; nothing algorithmic lives under this directory. Basilisk is the first simulation host by
ADR 0001 rev 1. `gnc-algorithms` is an editable path dependency on a sibling `WiseGncDesign`
checkout, built fresh from whatever revision sits beside this repository.

## Layout

- `models/` — reusable, bskSim-style dynamics stacks (spacecraft, gravity, sensors); the plant,
  carrying no algorithm.
- `hosting/` — `AlgorithmModule`, one generic Basilisk `SysModel` that hosts any WiseGncDesign
  algorithm through declarative field maps; the seam, carrying no vocabulary belonging to any
  one algorithm, sensor, or dynamics model, enforced by `scripts/check-hosting-generic.sh`.
- `scenarios/` — one `test_<scenario>.py` module per study, each a parametrized pytest module
  that wires a `models/` plant to an `AlgorithmModule` binding (declared in `bindings.py`) and
  owns its own dispersions and contract-export column map. A single nominal case is
  `execute_scenario(create_scenario())`, a plain function call; a campaign is this directory run
  as pytest, `pytest sim/basilisk/scenarios --runs N --out DIR [-n WORKERS]`.
- `conftest.py` — the campaign's `--runs`/`--out`/`--seed`/`--duration-s` options and the
  `plant`/`algorithm_binding`/`draws` fixtures every study's cases are built from, plus the
  session-end fold of every case's exported run into one contract v0 manifest.

## Exported sensor columns

`gyro_*`/`accel_*` in an exported run file carry the IMU's reading in its own platform frame,
which equals the body frame only when the sensor's mounting is identity, as in this demo. The
t=0 s row of every exported run reads zero for the IMU columns: Basilisk writes a sensor's first
output after the first dynamics step, not at `Reset()`.
