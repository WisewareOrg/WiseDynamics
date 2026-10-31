# Basilisk

Basilisk is a locked dependency (`bsk[all]`) of this repository's uv project, installed from prebuilt
wheels. Scenarios live here, importing GNC algorithms
from [WiseGncDesign](https://github.com/WisewareOrg/WiseGncDesign) rather than reimplementing
them; nothing algorithmic lives under this directory. Basilisk is the first simulation host by
ADR 0001 rev 1. `gnc-algorithms` is an editable path dependency on a sibling `WiseGncDesign`
checkout, built fresh from whatever revision sits beside this repository.

## Layout

- `hosting/` — `AlgorithmModule`, one generic Basilisk `SysModel` that hosts any WiseGncDesign
  algorithm through declarative field maps; it carries no vocabulary belonging to any one
  algorithm, sensor, or dynamics model, enforced by `scripts/check-hosting-generic.sh`.
- `models/` — reusable, bskSim-style dynamics stacks (spacecraft, gravity, sensors); a dynamics
  model carries no algorithm.
- `scenarios/` — one file per demo scenario, each wiring a `models/` dynamics stack to an
  `AlgorithmModule` instance for one specific algorithm.
- `campaigns/` — Monte Carlo campaign drivers and exporters over a scenario.

## Exported sensor columns

`gyro_*`/`accel_*` in an exported run file carry the IMU's reading in its own platform frame,
which equals the body frame only when the sensor's mounting is identity, as in this demo. The
t=0 s row of every exported run reads zero for the IMU columns: Basilisk writes a sensor's first
output after the first dynamics step, not at `Reset()`.
