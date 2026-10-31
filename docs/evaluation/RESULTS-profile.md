# Basilisk (`bsk`) wall-time profile — follow-on to RESULTS.md and RESULTS-orbit.md

Follow-on to `RESULTS.md` and `RESULTS-orbit.md`. Same environment: venv at `.venv` (bsk 2.12.1,
Python 3.13), examples sparse-checked-out at `basilisk-src/examples` (tag `v2.12.1`), 8 cores,
`MPLBACKEND=Agg`, `show_plots=False`. Every timing is `time.perf_counter()` around the relevant
call(s), via driver scripts in `scripts/`. All new scenario/model copies live in `scripts/`; the
sparse-checked-out `basilisk-src/` originals were never modified. Multiprocessing scenario modules
are imported by their real on-disk module name with their directory on `sys.path` (reusing
`scripts/run_mc_orbit.py` from the orbit follow-on), never via a synthetic `importlib.util`
module name, or pickling fails inside worker processes.

**Baseline:** `scripts/scenario_FeedbackRW_orbit.py` (bskSim stack, one 166-minute/9960 s orbit).
Re-measured here with the same driver pattern as the ablations (`scripts/run_ablation.py`, 3 runs,
median): **11.650 s** (runs: 10.979, 11.650, 11.975 s; logs `logs/4x-10-baseline-run{1,2,3}.log`).
This is consistent with the previously recorded 10.816 s / "~10.8 s" figure; the ~1 s difference is
run-to-run variance on this host, not a methodology change.

## Step and rate facts (BskSim dynamics/FSW stacks)

Read from `basilisk-src/examples/BskSim/BSK_masters.py`, `models/BSK_Dynamics.py`, and
`models/BSK_Fsw.py`:

- **`BSKSim.__init__(self, fswRate=0.1, dynRate=0.1)`** (`BSK_masters.py` line 37) — the baseline
  scenario calls `super().__init__()` with no arguments, so both rates default to **0.1 s**.
- **Dynamics step (`dynRate`):** `BSK_Dynamics.BSKDynamicModels.__init__(self, SimBase, dynRate)`
  sets `self.processTasksTimeStep = mc.sec2nano(dynRate)` and creates a single dynamics task
  (`"DynamicsTask"`) at that step. With `dynRate = 0.1 s` shipped.
- **FSW step (`fswRate`):** `BSK_Fsw.BSKFswModels.__init__(self, SimBase, fswRate)` sets
  `self.processTasksTimeStep = mc.sec2nano(fswRate)` and creates **9** FSW tasks at that same step
  (`inertial3DPointTask`, `hillPointTask`, `sunSafePointTask`, `velocityPointTask`,
  `mrpFeedbackTask`, `mrpSteeringRWsTask`, `mrpFeedbackRWsTask`, and two Lambert-guidance tasks); only
  the tasks for the active mode are enabled at a time, but all 9 are still scheduled/checked every
  step. With `fswRate = 0.1 s` shipped (same as `dynRate`).
- **Integrator:** `BSK_Dynamics.py` never calls `setIntegrator(...)` on `scObject`, so the
  spacecraft uses Basilisk's `DynamicObject` base-class default integrator, **RK4**
  (`svIntegratorRK4`), confirmed by inspecting the module's available integrator classes
  (`Basilisk.simulation.svIntegrators`) and the absence of any override in the dynamics model.
- **Recorder sampling:** `scenario_FeedbackRW_orbit.py`'s `log_outputs()` sets
  `samplingTime = FswModel.processTasksTimeStep` (i.e. the **fswRate**, 0.1 s) and uses it for all
  **7** recorders it adds to the dynamics task: `rwSpeedRec`, `rwMotorRec`, `attErrRec`, and 4×
  per-wheel `rwLogs[i]`.
- **Step counts for one 9960 s (166 min) orbit**, all confirmed directly against the cProfile call
  counts below (`StepUntilStop` called exactly 99600 times; `checkEvent` called exactly 996000 =
  99600 × 10 times, matching the 1 dynamics + 9 FSW tasks all being checked every dynamics step):
  - Dynamics steps: 9960 s / 0.1 s = **99,600**.
  - FSW task checks: 99,600 steps × 10 tasks (1 dyn + 9 fsw) = **996,000** `checkEvent` calls.
  - Recorder samples: sampled at the 0.1 s fswRate ⇒ ≈99,601 samples per recorder × 7 recorders ≈
    **697,207** total recorded samples for the one-orbit baseline run.

## Part A — profiling the baseline once

Profiled with `cProfile`, phase-separated (`scripts/profile_baseline.py` uses three separate
`cProfile.Profile()` objects — one each for construction, Initialize+ExecuteSimulation, and
pull_outputs — merged afterward with `pstats.Stats(...)` for one combined top-25-by-cumulative-time
list; this avoids a measurement artifact seen when a single profiler spans all three phases, see
"Profiling artifact" note below). Log: `logs/4x-00-profile_baseline.log`.

### Three-way split, in seconds and percent of the profiled run

| Phase | Seconds (profiled) | % of profiled total | Seconds (unprofiled, `logs/4x-01-time_baseline_split.log`) | % of unprofiled total |
|---|---|---|---|---|
| Scenario construction (module instantiation, SPICE kernel loading, gravity body setup) + `InitializeSimulation` | 0.0657 | 0.50% | 0.0513 | 0.49% |
| `ExecuteSimulation` (both calls: 10 min `inertial3D` + rest of orbit `directInertial3D`) | 11.9045 | 90.48% | 9.7073 | 93.29% |
| Post-run data retrieval and plotting (`pull_outputs`) | 1.1854 | 9.01% | 0.6483 | 6.23% |
| **Total** | **13.1557** | 100% | **10.4069** | 100% |

Choice made to get a measurement: `InitializeSimulation()` (0.0012 s, negligible) is folded into the
construction bucket rather than the execute bucket, since it is message-routing/setup work, not the
step loop itself; it is 2 orders of magnitude smaller than either neighboring bucket either way.

`ExecuteSimulation`/the step loop dominates under both measurements (90–93% of wall time); profiling
instrumentation itself inflates total wall time from ~10.4 s to ~13.2 s (cProfile's per-call
overhead, mostly paid inside the 99,600-call step loop and the matplotlib-heavy plotting code), and
shifts a couple of points of share from execute to post-run (plotting is unusually call-heavy per
unit of real work, so profiler overhead hits it harder proportionally) — but the ranking of the three
buckets is unchanged.

### Top 25 entries by cumulative time (verbatim, combined/merged profile)

```
         8039328 function calls (8028146 primitive calls) in 13.253 seconds

   Ordered by: cumulative time
   List reduced from 2603 to 25 due to restriction <25>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    1.721    0.860    9.260    9.260 .../cpython-3.13.15-linux-x86_64-gnu/lib/python3.13/threading.py:641(wait)
    99600    0.036    0.000    8.110    0.000 .../Basilisk/architecture/sim_model.py:1523(StepUntilStop)
    99600    8.074    0.000    8.074    0.000 {built-in method Basilisk.architecture._sim_model.SimModel_StepUntilStop}
      3/2    0.488    0.163    2.641    1.320 .../Basilisk/utilities/SimulationBaseClass.py:1936(ExecuteSimulation)
     61/4    0.010    0.000    1.087    0.272 .../matplotlib/text.py:926(get_window_extent)
   996001    0.769    0.000    0.929    0.000 .../Basilisk/utilities/SimulationBaseClass.py:200(checkEvent)
     57/1    0.002    0.000    0.537    0.537 .../matplotlib/text.py:358(_get_layout)
    235/2    0.001    0.000    0.533    0.267 .../matplotlib/text.py:65(_get_text_metrics_with_cache)
        1    0.063    0.063    0.366    0.366 .../BskSim/plotting/BSK_Plotting.py:166(plot_attitude_error)
        1    0.000    0.000    0.316    0.316 .../Basilisk/utilities/SimulationBaseClass.py:1632(_drawModuleMessageNodes)
       30    0.001    0.000    0.278    0.009 .../Basilisk/utilities/SimulationBaseClass.py:1691(_drawModulePorts)
      112    0.000    0.000    0.275    0.002 .../matplotlib/_api/deprecation.py:442(wrapper)
      112    0.001    0.000    0.274    0.002 .../matplotlib/__init__.py:1521(inner)
      112    0.003    0.000    0.272    0.002 .../matplotlib/axes/_axes.py:4782(scatter)
    99625    0.126    0.000    0.258    0.000 .../numpy/linalg/_linalg.py:2599(norm)
   996000    0.167    0.000    0.167    0.000 .../Basilisk/utilities/SimulationBaseClass.py:214(nextCheckTime)
      112    0.002    0.000    0.151    0.001 .../matplotlib/axes/_base.py:2349(add_collection)
       96    0.001    0.000    0.136    0.001 .../matplotlib/axes/_base.py:2484(add_patch)
   996001    0.134    0.000    0.134    0.000 .../Basilisk/utilities/SimulationBaseClass.py:184(shouldBeChecked)
1216806/1216283 0.126    0.000    0.126    0.000 {built-in method builtins.len}
       96    0.003    0.000    0.124    0.001 .../matplotlib/axes/_base.py:2497(_update_patch_limits)
        1    0.000    0.000    0.120    0.120 .../Basilisk/utilities/SimulationBaseClass.py:1552(_drawMessageEdges)
  807/129    0.003    0.000    0.097    0.001 .../matplotlib/axes/_base.py:887(_unstale_viewLim)
       16    0.000    0.000    0.095    0.006 .../matplotlib/pyplot.py:3830(plot)
  114/113    0.030    0.000    0.095    0.001 .../matplotlib/axes/_base.py:2962(autoscale_view)
```
(paths abbreviated with `...` for readability; full paths are in `logs/4x-00-profile_baseline.log`)

**Profiling artifact, noted explicitly:** the `threading.py:641(wait)` entry (9.26 s cumulative, 2
calls) is **not** 9 seconds of serialized lost time. It sits inside the `ExecuteSimulation`-phase
profiler and overlaps with (does not add to) `StepUntilStop`'s 8.074 s — both are inside the same
11.90 s execute-phase wall-clock window measured by `perf_counter`. It is consistent with
Basilisk's console progress-bar background thread (`simulationProgessBar.py`, visible further down
the profile) being joined at the end of `ExecuteSimulation`; cProfile's thread accounting makes a
concurrent background thread's wait look like large "cumulative time" on the call it's nested
under, even though it is idle/overlapping wall time, not additional serialized cost. This does not
change the conclusion: `SimModel_StepUntilStop` (the C++ step loop behind the SWIG call), at 8.074 s
of real `tottime` over 99,600 calls, is the single largest unambiguous, attributable cost in the run.

## Part B — ablations (one change each, 3 runs, median wall, full 166 min orbit)

Driver: `scripts/run_ablation.py` (construction + `runScenario` + `pull_outputs`, same three-phase
split as Part A, plus `FINAL_SIM_TIME_S` to confirm the full orbit ran and
`FINAL_SIGMA_BR_NORM` — the norm of the last recorded `attGuidMsg.sigma_BR` row — as the
end-of-run attitude-error scalar). All 6 ablations completed the full 9960 s orbit on all 3 runs;
zero warnings/errors/tracebacks in any of the 21 ablation+baseline run logs
(`logs/4x-10-*.log`; `grep -i "warn\|error\|exception\|traceback"` found nothing beyond the
literal, expected `FAILURES=[]` substring match on "fail").

| # | Ablation | Changed | Wall s (median of 3) | Speedup vs baseline (11.650 s) | End-of-run \|σ_BR\| |
|---|---|---|---|---|---|
| — | Baseline (`scenario_FeedbackRW_orbit.py`) | — | 11.650 | 1.00× | 6.641e-13 |
| 1 | dynRate → 0.5 s, fswRate unchanged (0.1 s) | 1 line (`BSKSim.__init__` call) | 4.908 | 2.37× | 4.684e-13 |
| 2 | dynRate → 1.0 s, fswRate also → 1.0 s (fswRate 0.1 s was finer than 1.0 s, so both raised per instructions) | 1 line | 1.583 | 7.36× | 4.817e-12 |
| 3 | Magnetic field (`magneticFieldWMM`) + magnetometer (`magnetometer`) removed from the dynamics model | ~9 lines across 3 edit sites, in a dynamics-module copy (`scripts/BSK_Dynamics_nomag.py`) | 10.613 | 1.10× | 6.641e-13 |
| 4 | SPICE "smaller version": kept SPICE, dropped sun and moon as gravity bodies (earth only); also dropped the eclipse and CSS-constellation modules, which depend on the sun gravity body | ~11 lines across 3 edit sites, in a dynamics-module copy (`scripts/BSK_Dynamics_nospice3body.py`) | 10.243 | 1.14× | 6.641e-13 |
| 5 | All 7 recorders thinned from 1 sample/fswRate step (0.1 s) to 1 sample/60 s | 1 line (`samplingTime` assignment) | 11.156 | 1.04× | 6.641e-13 |
| 6 | Combined: ablation 2 (dynRate/fswRate → 1.0 s, chosen over ablation 1 as the more aggressive still-working rate change) + ablation 3 + ablation 4 + ablation 5, all together | reuses the same per-ablation edits, composed in one dynamics-module copy (`scripts/BSK_Dynamics_combined.py`) + one scenario copy | 1.126 | 10.35× | 4.817e-12 |

Ablation 4, full-version note: the task's full version (SPICE replaced with the gravity factory's
non-SPICE path / `planetEphemeris`, dropping sun+moon gravity and the eclipse module) was skipped as
too invasive for the ~20-line budget once done properly. Removing `createSpiceInterface()` outright
would also require re-wiring `EarthEphemObject.addSpiceInputMsg(...)` (which specifically expects a
SPICE planet-state message) and everything keyed off `self.epochMsg` (propagated to the magnetic
field model's WMM date subscription and other dynamics submodules) — i.e. it touches the
message-passing contract several dynamics submodules rely on, not just gravity-body creation. The
smaller version (keep SPICE, drop sun/moon as gravity bodies, and drop the sun-dependent
eclipse/CSS wiring that would otherwise break on the missing sun body) stayed at ~11 changed lines
and is what is measured above.

All six ablations converge to a tiny end-of-run attitude error (σ_BR norm of order 1e-12–1e-13,
same order as the baseline's 6.64e-13), confirming none of them broke the control loop. The two
rate ablations (1, 2, and 6 which includes 2) show a slightly larger — but still minuscule —
residual error, consistent with a coarser RK4 step size producing a slightly less tightly converged
attitude at the same simulated end time, not a loss of convergence.

Dominant finding: the dynamics/FSW step **rate** is overwhelmingly the largest lever (2.4–7.4× alone,
10.3× combined); removing the magnetic field/magnetometer, dropping sun/moon gravity + eclipse/CSS,
and thinning recorders each save only 4–14% individually, and their combined effect on top of the
rate change is small (1.583 s → 1.126 s, ablation 2 alone to ablation 6) because the step loop they
would otherwise lighten has already been cut 10× by the rate change.

## Part C — Monte Carlo overhead (16 runs, 8 workers)

Base script: `scripts/scenarioBskSimAttFeedbackMC_orbit.py` (as already set up in the orbit
follow-on, pointing at `scripts/scenario_AttFeedback_orbit.py`), overridden to 16 runs / 8 workers
via `scripts/run_mc_orbit.py scripts/scenarioBskSimAttFeedbackMC_orbit.py 16 8`.

Instrumentation (`scripts/scenario_AttFeedback_orbit.py`, both edited in place since it's already a
`scripts/`-local copy, not a `basilisk-src/` original): `time.perf_counter()` stamps around (a) the
`scenario_AttFeedback.__init__` body — the "scenario construction function" the MC `Controller`
calls via `setSimulationFunction` — recorded as `self._construction_seconds`; and (b) inside
`runScenario(scenario)` — the "execution function" the `Controller` calls via
`setExecutionFunction` — split into **execution** (`InitializeSimulation` + `ExecuteSimulation`) and
**retention/data-pull**. The data-pull step replicates, read-only, exactly what
`RetentionPolicy.getDataForRetention` does right after `runScenario` returns (reading
`scenario.msgRecList["attGuidMsg"].sigma_BR`/`.omega_BR_B` and
`scenario.msgRecList["sNavTransMsg"].r_BN_N` via `simHelpers.addTimeColumn`), since the real pull
happens inside `Basilisk`'s installed `Controller.py`, which this task does not modify; this choice
(an equivalent, timed, read-only replica rather than patching the installed package) is made to get
a real measurement of that step's cost without touching code outside `scripts/`. Each worker process
appends one line per run to `logs/4x-20-mc_per_run.log` (baseline campaign) /
`logs/4x-21-mc_per_run_combined.log` (combined-ablation campaign, see below).

All 16 runs succeeded in both campaigns (`FAILURES=[]` in both `logs/4x-20-mc_baseline_16x8.log`
and `logs/4x-21-mc_combined_16x8.log`); no warnings/errors/tracebacks in either log.

### Baseline MC campaign (16 runs / 8 workers, `scenario_AttFeedback_orbit.py`)

| Metric | Value |
|---|---|
| Mean construction s (16 runs) | 0.1124 |
| Mean execution s (16 runs) | 32.2510 |
| Mean retention/data-pull s (16 runs) | 0.0853 |
| Total wall s | 73.6176 |
| Single-run baseline (Part A/B `scenario_FeedbackRW_orbit.py`, median) | 11.650 s |
| Ideal wall if perfectly parallel: 16 × 11.650 / 8 | 23.300 s |
| **Ratio actual/ideal** | **3.160×** |

### Combined-ablation MC campaign (16 runs / 8 workers, `scripts/scenarioBskSimAttFeedbackMC_orbit_combined.py` → `scripts/scenario_AttFeedback_orbit_combined.py`)

Ran because ablations 3 and 4 both worked (per instructions: "if ablation 4 or 3 worked..."). This
campaign uses the combined-ablation scenario (ablation 6's recipe: rate → 1.0 s, no magnetic
field/magnetometer, earth-only gravity with eclipse/CSS dropped, recorders thinned to 60 s), applied
to the MC scenario class the same way ablation 6 applied it to the single-run baseline.

| Metric | Value |
|---|---|
| Mean construction s (16 runs) | 0.0920 |
| Mean execution s (16 runs) | 1.6465 |
| Mean retention/data-pull s (16 runs) | 0.0003 |
| Total wall s | 5.7609 |
| Ratio actual/ideal, same denominator as above (16 × 11.650 / 8 = 23.300 s) | 0.247× |
| Ratio actual/ideal using the combined-ablation single-run median instead (16 × 1.126 / 8 = 2.252 s) | 2.559× |

The "ratio actual/ideal" denominator choice: the task asks for the ratio against "16 × single-run
baseline / 8" — read here as the named Baseline from Part A/B (`scenario_FeedbackRW_orbit.py`),
kept as the same fixed denominator for both campaigns so the two ratios are directly comparable and
show the combined campaign's total improvement against the one reference point used throughout this
report. The second ratio (using the combined-ablation scenario's own single-run median as the
denominator) is reported in addition, as the more apples-to-apples parallel-efficiency figure for
that scenario specifically.

## Where the time goes

For a single 166-minute-orbit run, `ExecuteSimulation` — the C++ step loop behind the SWIG call —
accounts for 90–93% of wall time (measured both under cProfile and unprofiled); scenario
construction (module instantiation, SPICE kernel loading, gravity body setup) is under 1%; post-run
data retrieval and plotting is the remainder (6–9%, more under profiling because plotting code is
unusually call-heavy and profiler overhead scales with call count). Within the step loop, the
dynamics/FSW step rate is the dominant lever: raising `dynRate`/`fswRate` from the shipped 0.1 s to
1.0 s (a 10× reduction in step count) cuts wall time 7.36× alone and 10.35× combined with every
other ablation, while removing the magnetic field model and magnetometer, dropping sun/moon gravity
plus the sun-dependent eclipse/CSS modules, and thinning recorders to 1 sample/60 s each save only
4–14% individually and little more once the rate change is already applied — because those per-step
costs are a small fraction of a step whose count has already been cut 10×. In the 16-run/8-worker
Monte Carlo campaign, construction (0.11 s mean) and retention/data-pull (0.09 s mean) are both
negligible next to execution (32.25 s mean, 99.4% of per-run time); the campaign's 73.62 s total
wall time is 3.16× the wall time 16 runs would take if perfectly parallelized across 8 workers at
the single-run baseline's rate, and that same ~3.16× factor reappears almost exactly
(32.25 s mean execution under 8-way contention ÷ 10.18 s for one uncontended run of the same
scenario) as the per-run slowdown from 8 processes sharing 8 physical cores — i.e. the Monte Carlo
"overhead" above ideal scaling is explained by CPU contention during execution, not by
construction, retention, or any fixed per-run dispatch cost. Running the same 16-run campaign on
the combined-ablation scenario reproduces this ~2.56× contention ratio against its own (much
smaller) single-run cost, while cutting total campaign wall time from 73.62 s to 5.76 s (12.78×).

## Warnings / errors verbatim

None. `grep -i "warn\|error\|exception\|traceback"` over every log produced in this follow-on
(`logs/4x-00-*.log` through `logs/4x-21-*.log`) found no genuine warning, error, exception, or
traceback; the only matches were the literal substring "fail" inside the expected `FAILURES=[]`
lines and inside the filename `BSK_Plotting.py` (`plot_attitude_error`), both false positives. One
harmless, expected `UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown` was
seen only when `scripts/scenario_AttFeedback_orbit.py` was smoke-tested directly with
`show_plots=True` under the `Agg` backend (not part of any reported measurement, since the MC
campaigns and all Part A/B drivers call with `show_plots=False`).

## Files

- New driver scripts: `scripts/profile_baseline.py` (Part A), `scripts/time_baseline_split.py`
  (Part A cross-check), `scripts/run_ablation.py` (Part B, generic single-scenario runner).
- Ablation scenario copies (`scripts/scenario_FeedbackRW_orbit_ablation{1..6}_*.py`), each a copy of
  the baseline `scripts/scenario_FeedbackRW_orbit.py` with exactly the one (or, for #6, composed)
  change described in the Part B table.
- Ablation dynamics-module copies: `scripts/BSK_Dynamics_nomag.py` (ablation 3),
  `scripts/BSK_Dynamics_nospice3body.py` (ablation 4), `scripts/BSK_Dynamics_combined.py`
  (ablations 3+4 together, used by ablation 6 and by the Part C combined-ablation MC campaign).
- Part C: `scripts/scenario_AttFeedback_orbit.py` (instrumented in place),
  `scripts/scenario_AttFeedback_orbit_combined.py` (instrumented, combined-ablation scenario),
  `scripts/scenarioBskSimAttFeedbackMC_orbit_combined.py` (MC driver pointed at the combined
  scenario, separate archive dir).
- Logs (4x- prefix): `logs/4x-00-profile_baseline.log`, `logs/4x-00b-profile_baseline_rerun.log`
  (reproducibility check), `logs/4x-01-time_baseline_split.log`,
  `logs/4x-10-baseline-run{1,2,3}.log`, `logs/4x-10-ablation{1_dyn05,2_dyn10,3_nomag,4_nospice,
  5_thinrec,6_combined}-run{1,2,3}.log`, `logs/4x-20-mc_baseline_16x8.log`,
  `logs/4x-20-mc_per_run.log`, `logs/4x-21-mc_combined_16x8.log`,
  `logs/4x-21-mc_per_run_combined.log`.
