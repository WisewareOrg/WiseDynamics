# 0003 — Define each fidelity by its integrator and an accuracy budget, stepping at the flight software's period

**Status:** accepted
**Decided:** 2026-11-12 (issue 28, ground process)
**Rev:** 2

## Revisions

- **rev 2** — 2026-11-12 — the recording period is dropped: truth and the IMU are recorded at every
  step, so a run refuses no recording period and the step follows the flight software's period
  alone (issue 28, ground process). The step-dividing-both alternative becomes the recording-period
  alternative.
- **rev 1** — 2026-11-12 — first written (issue 26, fidelity tiers).

## Context

A fidelity was a fixed step and integrator: `hi` RK4 at 0.02 s, `med` RK4 at 0.05 s, `lo` RK2 at
0.1 s. The step stood in for accuracy, so nothing stated how far a fidelity's truth could be from
the exact solution, and the cost of a run did not follow the flight software's rate. A 852-case
Monte Carlo of the 1 Hz flight software on Leo500 at `lo` took about 9 minutes on a 4-core laptop,
too slow to iterate on algorithms.

The IMU ran in the dynamics task, so it sampled every step. A case's noise realization changed
with its fidelity (gyro outputs of one case differed by 1.9e-4 rad/s between fidelities, against
5e-5 rad/s of noise), so two fidelities of a case could not be compared for numerical error alone.

WiseDynamics knows one thing about the flight software under test, its `period_s`. A fidelity has
to be defined without anything more.

## Decision

A fidelity is an integrator and a budget on its truth position's distance from `hi`'s over one
orbit, which a test holds every scenario to:

| fidelity | integrator | budget |
|---|---|---|
| `hi` | RKF78, relative tolerance 1e-10, absolute 1e-12 | the reference |
| `med` | RK4 | 1 mm |
| `lo` | Ralston's third-order Runge-Kutta | 1 m |

Every fidelity steps at the largest step of at most 1 s that divides the flight software's period.
Truth and the IMU are recorded at every step. The IMU and the flight software's own fields change
only at its period, so their rows hold the last write between its ticks. A run refuses, before
anything runs, a flight-software period that no step of at least 1 ms divides. The 1 s maximum is a
chosen bound, not a measured one. A vehicle's sensors run in their own task at the flight
software's period, after the dynamics and before the flight software, so a case draws the same
sensor noise at every fidelity. A sensor's white noise is a density; a sample carries 1-sigma
density / sqrt(period).

## Alternatives considered

Measured on the IMU spacecraft through WiseGncDesign's dead-reckoner Monte Carlo: 460 dispersed
cases of one orbit of Leo500 and Leo400, a 1 Hz Python flight software, the IMU sampled at its
period so every row draws the same noise, on a 4-core laptop. Each accuracy is the worst over the
cases, against RKF78 at 1 s at Basilisk's default tolerances. The batch times are that machine's,
which drifted by up to 40% between identical runs. `tests/test_fidelity.py` re-derives each
fidelity's accuracy on the nominal scenarios.

| integrator, step | batch | truth position, worst | truth attitude, worst |
|---|---|---|---|
| RK2, 0.1 s (former `lo`) | 179 s | 0.71 m | 86 urad |
| RK2, 1 s | 104–105 s | 73 m | 8.7 mrad |
| Ralston's RK3, 1 s | 106 s | 6.6 mm | 64 urad |
| RK4, 1 s | 99 s | 1 um | 0.085 urad |
| RK4, 0.1 s | 270 s | 1.5 um | 0.035 nrad |
| RKF78, 1 s | 238–247 s | (reference) | (reference) |
| RK4, 0.02 s (former `hi`), 40 cases | 8.2 s per case | 0.6 um | 6 prad |

- **Fixed steps per fidelity, the former design.** A finer step bought accuracy no budget asked
  for: RK4 at 0.1 s is within 1.5 um of the reference and costs 2.7 times RK4 at 1 s. RK2 at 0.1 s
  was the only source of real error and was slower than RK4 at 1 s.
- **RK2 at 1 s for `lo`.** Half RK3's integration cost, but 73 m from the reference, outside the
  1 m budget.
- **RK4 for `lo` as well.** Rejected on cost: with the flight software at 0.1 Hz, so that the
  integration dominates a case, RK3 measured 23% faster than RK4 on the dynamics alone (0.065 s
  against 0.084 s an orbit, interleaved) and 12% faster on a 460-case Monte Carlo (28–31 s against
  34–36 s). At 1 Hz the flight software dominates and the difference is not measurable.
- **Adaptive integrators for `lo` or `med`.** Basilisk's adaptive integrators subdivide a task
  step, never lengthen it, so at a step pinned by the flight software they take the same step with
  more stages: RKF45 at 1 s costs 2.4 times RK4 and RKF78 5.4 times.
- **RK4 at 0.02 s for `hi`, the former reference.** RKF78 at 1 s agrees with it to 0.6 um at a
  quarter of its cost, and holds its error to its tolerance where a fixed step does not. Tightening
  RKF78 from Basilisk's default relative tolerance of 1e-4 to 1e-10 cost nothing measurable on
  Leo500 at a 1 s step.
- **A step chosen per fidelity from its budget**, longer than 1 s where the flight software's
  period allows. Not taken: the 1 s maximum is chosen, and every measurement here is at a step of
  at most 1 s.
- **A recording period.** A step dividing both the flight software's period and a recording period
  is silent where the two share no step of 1 s or less: a 3 Hz flight software recorded every 10 s
  steps at 1 ns, and a run never finishes. Dropped altogether: a recording period set on the
  simulation side decimates what the flight software writes, whose telemetry rate the flight
  software owns, and analyzers compare each write with truth at its own time. Truth is recorded at
  every step instead, and every flight-software tick is one.
- **Sensors in the dynamics task.** Splitting them out cost nothing measurable (coupled and split
  matched within run-to-run noise), and coupled sensors change a case's noise with its fidelity.

## Consequences

The 1 Hz Leo500 Monte Carlo runs about 1.8 times faster at `lo` or `med` than at the former `lo`,
and every fidelity's truth is closer to the reference than the former `lo`'s. What remains of a case
at 1 Hz is mostly the flight software. A slower flight software makes a case cheaper at every
fidelity; a faster one steps every fidelity finer.

A run file holds a row per step, so its size is the duration over the step, whatever the flight
software records. A faster flight software costs disk as well as time.

A fidelity's budget holds for the scenarios the test runs. A scenario or vehicle with stiffer
dynamics (faster spin, wheels, flexible modes) can break a budget at the 1 s step; the test then
fails rather than the results degrading.

A sensor samples only at the flight software's period. A sensor that samples faster than the flight
software reads it, and integrates or averages between reads, needs its own period, which this
design does not provide.

The configuration a run file's `fidelity` attribute names is the one of the WiseDynamics version
its `versions_wise_dynamics` attributes record. A vehicle's noise parameters are densities; the
per-sample standard deviation at a sample period is density / sqrt(period).
