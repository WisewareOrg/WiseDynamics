# 0001 — Simulate spacecraft dynamics with Basilisk as the first simulation host

**Status:** accepted
**Decided:** 2026-10-31 (evaluation on the development host; results in `docs/evaluation/`)
**Rev:** 1

## Revisions

- **rev 1** — 2026-10-31 — first written (#1 adopt Basilisk as the first simulation host).

## Context

WiseDynamics is the simulation home of a three-repository GNC project. Flight code lives in
WiseGncDesign and must not depend on any simulation tool. Analysis lives in WiseGncAnalysis and
reads recorded runs through a data contract, never a simulator's native output. WiseDynamics owns
every simulation plant, the wrappers that host WiseGncDesign algorithms in each plant, the Monte
Carlo drivers, and the exporters to the data contract. Which plants it supports is the one trade
this project has to make, and this ADR makes the first of them.

The constraints a first plant has to meet:

- Spacecraft rigid-body dynamics with the attitude hardware a vehicle actually carries (reaction
  wheels, thrusters, star tracker, sun sensors, IMU, magnetometer), supplied by the tool rather
  than written here.
- Monte Carlo campaigns as a first-class capability, dispersing initial conditions and model
  parameters and running on every core of the development host.
- Algorithm iteration in Python, with the algorithm under test running inside the plant's own
  step loop, so design and tuning do not wait on a C++ build.
- Throughput far beyond real time. A wall-clock-paced simulation is not acceptable in any
  configuration this project uses.
- Open source, installable on the development host (NixOS) without building from source.
- Proprietary, paid simulation tooling is out of scope for this project.

## Decision

Basilisk (AVS Lab, University of Colorado Boulder; PyPI package `bsk`, 2.12.x line) is the first
simulation host, at `sim/basilisk/`.

- Installed from prebuilt wheels with `pip install "bsk[all]"`.
- Scenarios are Python scripts. WiseGncDesign algorithms enter the loop through thin wrapper
  modules written here: a Basilisk Python module around a Python algorithm, a Basilisk C++ module
  around a C++ one. Nothing algorithmic lives in this repository.
- Campaigns use Basilisk's Monte Carlo controller and dispersion classes, in-process, with no
  clock pacing.
- Recorded runs are exported to the WiseGncAnalysis data contract; Basilisk's own retention
  format is never read by analysis.

Evidence from the evaluation (`docs/evaluation/RESULTS.md`, `RESULTS-orbit.md`): wheels installed
and imported on the development host without source builds; a ten-minute attitude scenario ran at
about 3200 times real time with a C controller and about 1300 times with the same controller as a
Python module; a 166-minute orbit of the full bskSim stack (four wheels, eight thrusters, eight sun
sensors, magnetometer, three-body gravity through SPICE, four guidance modes and a steering law)
ran at about 920 times real time; a 100-run Monte Carlo of that orbit completed on eight workers in
436 seconds with no failures. Profiling (`RESULTS-profile.md`) put 90 to 93 percent of a run in the
C++ step loop and showed the integration step as the dominant lever: the same orbit at a 1 s
dynamics and flight-software step ran in 1.1 s, and a 16-run campaign fell from 74 s to 5.8 s.

## Alternatives considered

- **Trick with JEOD and the Common Model Library (NASA JSC).** Strongest Monte Carlo and
  campaign infrastructure of any candidate, master-slave dispatch across machines, checkpointing,
  and an environment library (JEOD) with no open equal. Rejected for now because the plant has to
  be written: models are C++ and Python is the input language only, and the Common Model Library
  (released 2026-05, version 0.3.0 at the time of writing) carries RCS jets, rocket motors, aero,
  atmosphere and mass models but no reaction wheel, star tracker, IMU or sun sensor. This is the
  candidate most likely to become a second plant; see Consequences.
- **42 (NASA GSFC).** Attitude hardware models shipped, builds in seconds, pairs natively with cFS.
  Rejected: no Python interface, and Monte Carlo is a pair of driver scripts rather than a
  controller.
- **JSBSim.** Tabular six-degree-of-freedom atmospheric flight model with Python bindings.
  Rejected as a first plant: no orbital dynamics, no spacecraft actuator or sensor models, no
  Monte Carlo infrastructure. Remains the lightest route to a lifting-body entry plant if that
  work starts.
- **Orekit and other astrodynamics libraries (Tudat, nyx, poliastro).** Mission-analysis and
  orbit-propagation libraries, not plants with an actuator, sensor and flight-software seam.
- **Single-author Python attitude simulators found on GitHub (OpenGNC, OpenLodestar).** Immature,
  one contributor each, no validation lineage.
- **MATLAB and Simulink.** The tooling most of industry designs GNC in. Out of scope: proprietary
  and paid, which this project excludes by its own constraint above.

## Consequences

- An algorithm written as a Basilisk Python module costs roughly two and a half times the wall
  time of the same algorithm in C. Acceptable for design iteration; promotion to C++ in
  WiseGncDesign is the path when campaign throughput matters.
- Campaign throughput per worker was about one third of the single-run rate on the evaluation
  host. Per-run construction and data retention were each under 0.2 s, so the loss is CPU
  contention: eight workers on a four-core laptop processor with hyperthreading. The Monte Carlo
  harness adds no material fixed cost per run; campaign speed is set by step size and physical
  cores.
- The shipped bskSim examples integrate at 0.1 s. Scenarios here choose their step from the
  problem, not from the example, and state it.
- A plant that cannot run Python constrains WiseGncDesign: adding Trick would require a C++
  variant of every algorithm meant to run there. That cost is stated in the ADR that adds Trick.
- First use of a scenario that loads SPICE kernels or the World Magnetic Model downloads support
  data to a user cache (about 25 seconds once). CI that runs such scenarios needs the cache or
  network access.
- Vizard, Basilisk's visualizer, is a closed-source binary and optional. Nothing here depends on it.
- Revisit trigger: when the Common Model Library ships reaction wheel and star tracker models, or
  when guided-entry work for a lifting body begins, re-evaluate Trick with JEOD and CML as a
  second plant. That decision is a new ADR; this one is then revved only if Basilisk stops being
  the first host.
