# 0003 — Define each fidelity by its integrator and an accuracy budget, stepping at the flight software's period

**Status:** accepted
**Decided:** 2026-11-12 (issue 26, fidelity tiers, from the `spike/split-sensors` measurements)
**Rev:** 1

## Revisions

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

Every fidelity steps at the largest step of at most 1 s that divides the flight software's period
and the recording period. The 1 s maximum is a chosen bound, not a measured one. A vehicle's sensors
run in their own task at the flight software's period, after the dynamics and before the flight
software, so a case draws the same sensor noise at every fidelity. A sensor's white noise is a
density; a sample carries 1-sigma density / sqrt(period).

## Alternatives considered

Measured on the IMU spacecraft through a 460-case paired Monte Carlo of the 1 Hz Python flight
software, one orbit of Leo500 and Leo400, against RKF78 at 1 s:

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
  RKF78 from Basilisk's default relative tolerance of 1e-4 to 1e-10 cost nothing measurable.
- **A step chosen per fidelity from its budget**, longer than 1 s where the flight software's
  period allows. Not taken: the 1 s maximum keeps every fidelity's step within what was measured.
- **Sensors in the dynamics task.** Splitting them out cost nothing measurable (coupled and split
  matched within run-to-run noise), and coupled sensors change a case's noise with its fidelity.

## Consequences

The 1 Hz Leo500 Monte Carlo runs about 1.8 times faster at `lo` or `med` than at the former `lo`,
and every fidelity's truth is closer to the reference than the former `lo`'s. What remains of a case
at 1 Hz is mostly the flight software. A slower flight software makes a case cheaper at every
fidelity; a faster one steps every fidelity finer.

A fidelity's budget holds for the scenarios the test runs. A scenario or vehicle with stiffer
dynamics (faster spin, wheels, flexible modes) can break a budget at the 1 s step; the test then
fails rather than the results degrading.

A sensor samples only at the flight software's period. A sensor that samples faster than the flight
software reads it, and integrates or averages between reads, needs its own period, which this
design does not provide.

A run file's `fidelity` attribute means a different configuration before and after this change; the
file's `versions_wise_dynamics` attributes tell the two apart. A vehicle's noise parameters change
name and meaning: the per-sample standard deviation a consumer read is now density / sqrt(period).
