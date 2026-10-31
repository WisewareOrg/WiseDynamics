# 0001 — Simulate spacecraft dynamics with Basilisk as the first simulation host

**Status:** accepted
**Decided:** 2026-10-31 (issue #1 evaluation on the development host)
**Rev:** 2

## Revisions

- **rev 2** — 2026-10-31 — the sim is a library of plant models, a one-case runner, a run
  writer and a pytest plugin that depends on no flight software; consumers pin a tagged
  release; scenarios and campaigns live with the flight software they test (#6 add the Basilisk
  IMU dead-reckoning scenario and 852-run campaign).
- **rev 1** — 2026-10-31 — first written (#1 adopt Basilisk as the first simulation host).

## Context

WiseDynamics is a plant library for Wiseware's GNC work, depending on no flight software: a
consumer that hosts a named algorithm against one of its plants pins a tagged release of this
repository rather than this repository depending on that consumer. WiseDynamics owns every
simulation plant and the generic running/exporting mechanics (a one-case runner, a run writer, a
pytest plugin for Monte Carlo options and fixtures); a scenario or campaign that boots a named
flight-software instance lives with the flight software it tests. Which plants it supports is the
one trade this project has to make, and this ADR makes the first of them.

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

- Installed from prebuilt wheels as a locked dependency (`bsk[all]`) of this repository's uv
  project, the organisation's Python dependency manager. Never built from source here.
- `sim/basilisk/models/` holds static plant models (a spacecraft hub, its gravity, its sensors);
  `sim/basilisk/run.py::run_case` adds any flight-software module a consumer builds to a plant's
  task and records it; neither knows about, nor depends on, any particular algorithm.
- `sim/basilisk/pytest_plugin.py` is a pytest plugin, loaded by a consumer's own `conftest.py`
  (`pytest_plugins = [...]`): `--runs`/`--out`/`--seed`/`--duration-s`/`--spacecraft` options and
  `plant`/`draws`/`run_writer` fixtures, dispersed through plain `numpy.random`, run in-process
  with no clock pacing. A campaign is `pytest -n W --runs N` in the consumer's own test suite.
- `sim/basilisk/runs.py::write_run` writes one HDF5 run file per SPEC-demo.md §2's layout, given
  whatever arrays and attrs the caller supplies; Basilisk's own retention format is never read
  by analysis.

What the evaluation established: the wheels install and import on the development host without a
source build; a controller written as a Basilisk Python module runs inside the step loop alongside C
and C++ modules; a Monte Carlo campaign of a full-stack scenario (reaction wheels, thrusters, sun
sensors, magnetometer, three-body gravity, a multi-mode guidance chain) dispatches across every core
and completes with no failures; throughput is hundreds to thousands of times real time in every
configuration tried; and the integration step, not the model set, is what sets the cost of a run.

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

- An algorithm written as a Basilisk Python module costs a small multiple of the wall time of the
  same algorithm in C. Acceptable for design iteration; promotion to C++ in WiseGncDesign is the
  path when campaign throughput matters.
- pytest-xdist fans cases over workers; campaign speed is set by the integration step and
  physical cores.
- The shipped bskSim examples integrate at 0.1 s. Scenarios here choose their step from the
  problem, not from the example, and state it.
- A plant that cannot run Python constrains WiseGncDesign: adding Trick would require a C++
  variant of every algorithm meant to run there. That cost is stated in the ADR that adds Trick.
- First use of a scenario that loads SPICE kernels or the World Magnetic Model downloads support
  data to a user cache. CI that runs such scenarios needs the cache or network access.
- Vizard, Basilisk's visualizer, is a closed-source binary and optional. Nothing here depends on it.
- Revisit trigger: when the Common Model Library ships reaction wheel and star tracker models, or
  when guided-entry work for a lifting body begins, re-evaluate Trick with JEOD and CML as a
  second plant. That decision is a new ADR; this one is then revved only if Basilisk stops being
  the first host.
