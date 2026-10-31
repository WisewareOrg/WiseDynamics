# Basilisk (`bsk`) orbit-profile timing — follow-on to RESULTS.md

Follow-on to the evaluation in `RESULTS.md`. Same environment: venv at `.venv` (bsk 2.12.1,
Python 3.13), examples sparse-checked-out at `basilisk-src/examples` (tag `v2.12.1`), 8 cores,
`MPLBACKEND=Agg`, `show_plots=False`. Each timing is `time.perf_counter()` wrapped strictly
around the scenario's `run(...)` call (driver scripts in `scripts/`), not interpreter startup or
module import. Driver scripts: `scripts/run_feedbackrw3arg.py`, `scripts/run_module.py`,
`scripts/run_mc_orbit.py` (reuse the working pattern from `RESULTS.md`: the Monte Carlo scenario
module is imported by its real on-disk module name with its directory on `sys.path`, never via a
synthetic `importlib.util.spec_from_file_location` name, or multiprocessing pickling fails inside
worker processes).

## Orbital period computation

Both scenario families below use the same classical orbit elements:

```
oe.a = 10000000.0  m
oe.e = 0.01
oe.i = 33.3 deg
oe.Omega = 48.2 deg
oe.omega = 347.8 deg
oe.f = 85.3 deg
mu = earth.mu  (standard Earth gravitational parameter from Basilisk's gravBodyFactory)
```

`mu` was read directly from the running environment rather than assumed:

```
>>> from Basilisk.utilities import simIncludeGravBody
>>> gravBodyFactory().createEarth().mu
398600436000000.0   # m^3/s^2
>>> gravBodyFactory().createBodies(['sun','earth','moon'])['earth'].mu
398600436000000.0   # same value — confirms BskSim's BSK_Dynamics.py (createBodies) uses the
                     # identical constant as the single-file scenario (createEarth)
```

Orbital period (circular/elliptical two-body, only `a` and `mu` matter):

```
T = 2*pi*sqrt(a^3 / mu)
  = 2*pi*sqrt((1.0e7)^3 / 398600436000000.0)
  = 9952.014122896631 s
  = 165.86690204827718 min
```

Rounded up to the next whole minute: **166 min = 9960 s**. This value is used as the "one full
orbit" stop time everywhere below (both for the standalone scenario in step 1 and the BskSim
scenario/MC campaign in step 2, since they use the identical `a`/`mu`).

## Module inventory — BskSim dynamics and FSW stacks

Read from `basilisk-src/examples/BskSim/models/BSK_Dynamics.py` and `BSK_Fsw.py`, which every
BskSim-based scenario (`scenario_FeedbackRW.py`, `scenario_AttFeedback.py`,
`scenarioBskSimAttFeedbackMC.py`, etc.) builds on via `BSK_masters.BSKSim.set_DynModel`/
`set_FswModel`.

**Simulation (dynamics process) modules instantiated in `BSKDynamicModels.__init__`:**

| Attribute | Module class | Notes |
|---|---|---|
| `scObject` | `spacecraft.Spacecraft` | rigid body hub |
| `gravFactory` | `simIncludeGravBody.gravBodyFactory` | creates sun, earth, moon gravity bodies + SPICE `spiceObject` interface; earth is central body |
| `rwFactory` | `simIncludeRW.rwFactory` | builds 4x `Honeywell_HR16` reaction wheels (`RW1`–`RW4`) in `SetReactionWheelDynEffector` |
| `extForceTorqueObject` | `extForceTorque.ExtForceTorque` | external disturbance torque |
| `simpleNavObject` | `simpleNav.SimpleNav` | nav sensor (attitude/rate/position/velocity) |
| `eclipseObject` | `eclipse.Eclipse` | eclipse state vs. sun/other bodies |
| `CSSConstellationObject` | `coarseSunSensor.CSSConstellation` | 8x `coarseSunSensor.CoarseSunSensor` units (`CSS1`–`CSS8`) |
| `rwStateEffector` | `reactionWheelStateEffector.ReactionWheelStateEffector` | RW cluster effector on the hub |
| `thrustersDynamicEffector` | `thrusterDynamicEffector.ThrusterDynamicEffector` | 8x `MOOG_Monarc_1` ACS thrusters (`simIncludeThruster.thrusterFactory`) |
| `EarthEphemObject` | `ephemerisConverter.EphemerisConverter` | SPICE ephemeris to Basilisk message conversion |
| `magModule` | `magneticFieldWMM.MagneticFieldWMM` | World Magnetic Model field |
| `TAM` | `magnetometer.Magnetometer` | magnetometer sensor reading `magModule` |

**Flight-software modules instantiated in `BSKFswModels.__init__`:**

| Attribute | Module class | Notes |
|---|---|---|
| `inertial3D` | `inertial3D.inertial3D` | inertial-pointing guidance |
| `hillPoint` | `hillPoint.hillPoint` | Hill-frame pointing guidance |
| `sunSafePoint` | `sunSafePoint.sunSafePoint` | sun-safe pointing guidance |
| `velocityPoint` | `velocityPoint.velocityPoint` | velocity-vector pointing guidance |
| `cssWlsEst` | `cssWlsEst.cssWlsEst` | CSS-based sun direction estimator (weighted least squares) |
| `trackingError` | `attTrackingError.attTrackingError` | attitude tracking error vs. reference |
| `mrpFeedbackControl` | `mrpFeedback.mrpFeedback` | MRP feedback control (direct torque path) |
| `mrpFeedbackRWs` | `mrpFeedback.mrpFeedback` | second instance, MRP feedback control (RW path, feeds forward RW speeds) |
| `mrpSteering` | `mrpSteering.mrpSteering` | MRP steering law |
| `rateServo` | `rateServoFullNonlinear.rateServoFullNonlinear` | nonlinear rate servo loop paired with `mrpSteering` |
| `rwMotorTorque` | `rwMotorTorque.rwMotorTorque` | maps control torque to RW motor torques |
| `lambertPlannerObject` | `lambertPlanner.LambertPlanner` | Lambert transfer planner |
| `lambertSolverObject` | `lambertSolver.LambertSolver` | Lambert problem solver |
| `lambertValidatorObject` | `lambertValidator.LambertValidator` | validates/executes planned DV burn |
| `lambertSurfaceRelativeVelocityObject` | `lambertSurfaceRelativeVelocity.LambertSurfaceRelativeVelocity` | surface-relative velocity for second DV |
| `lambertSecondDvObject` | `lambertSecondDV.LambertSecondDV` | second Lambert DV burn computation |

FSW tasks are grouped by pointing mode (`inertial3DPointTask`, `hillPointTask`,
`sunSafePointTask`, `velocityPointTask`, `mrpFeedbackTask`, `mrpSteeringRWsTask`,
`mrpFeedbackRWsTask`, plus two Lambert-guidance tasks); only one guidance mode's tasks are enabled
at a time via `createNewEvent`/`modeRequest`. The scenarios run below use the `inertial3D` /
`directInertial3D` modes (`inertial3DPointTask` + `mrpFeedbackRWsTask`, or `mrpFeedbackTask`).

## Runs

| Scenario | Runs | Workers | Per-run simulated s | Total simulated s | Wall s | Ratio (total-sim/wall) |
|---|---|---|---|---|---|---|
| `scenarioAttitudeFeedbackRW.py` (as shipped) | 1 | 1 | 600 | 600 | 0.373771 | 1605.3 |
| `scenarioAttitudeFeedbackRW.py` (1 orbit, 166 min) | 1 | 1 | 9960 | 9960 | 2.800716 | 3556.2 |
| `scenario_FeedbackRW.py` (as shipped, cached) | 1 | 1 | 1800 | 1800 | 2.650712 | 679.1 |
| `scenario_FeedbackRW.py` (1 orbit, 166 min) | 1 | 1 | 9960 | 9960 | 10.816157 | 920.8 |
| `scenarioBskSimAttFeedbackMC.py` (as shipped) | 4 | 4 (effective; see note) | 600 | 2400 | 2.269258 | 1057.6 |
| `scenarioBskSimAttFeedbackMC_orbit.py` (100 runs, 8 workers, 1 orbit each) | 100 | 8 | 9960 | 996000 | 435.732685 | 2285.8 |

Choices made to get a measurement (noted per instructions, since the task was ambiguous on these
points):

- **Step 2a scenario choice:** ran `scenario_FeedbackRW.py` (not `scenario_AttGuidance.py`) — it
  is the direct BskSim analogue of `scenarioAttitudeFeedbackRW.py` used in step 1 (same orbit
  elements, same RW-feedback control story), and it ran cleanly.
- **`scenario_FeedbackRW.py`'s "simulated seconds":** `runScenario()` calls
  `scSim.ConfigureStopTime(...)` twice — first at 10 min (`inertial3D` mode), then at 30 min
  (`directInertial3D` mode). `ConfigureStopTime` sets an *absolute* stop time, not an incremental
  duration, so the total simulated duration for the as-shipped run is 30 min (1800 s), not
  10+30=40 min. For the one-orbit variant, only the **second** (final) stop time was raised, to
  166 min — the mode switch to `directInertial3D` still happens at the original 10-minute mark, and
  the run then continues in that mode out to the full orbital period. This was a deliberate choice:
  the alternative (raising both phases proportionally) was not that the instructions specified
  either way, so the simplest change that yields "one full orbit" of total simulated time was used.
- **First-run SPICE/WMM download overhead:** the very first execution of any BskSim scenario in
  this environment downloads SPICE kernels and the WMM magnetic-field coefficient file to
  `~/.cache/bsk_support_data` (one-time; not re-downloaded afterward). The first
  `scenario_FeedbackRW.py` run measured 25.733420 s wall, almost entirely download time (see
  `logs/3x-20-scenario_FeedbackRW-asshipped.log`); the table above reports the immediately-following
  cached re-run (`logs/3x-20b-scenario_FeedbackRW-asshipped-cached.log`, 2.650712 s) as the
  representative "as shipped" wall time, since the downloads are a one-time environment-setup cost
  unrelated to simulation throughput. The raw first-run number is kept in the log for the record.
- **MC worker count, as shipped:** `scenarioBskSimAttFeedbackMC.py` sets `setExecutionCount(4)` and
  leaves `setThreadCount(...)` commented out, so the `Controller` defaults to
  `mp.cpu_count()` = 8 threads, then logs `Fewer MCs spawned than processes assigned (4 < 8).
  Changing processes count to 4.` — same pattern already documented in `RESULTS.md` for
  `scenarioMonteCarloAttRW.py`. Effective worker count recorded above is 4.
- **100-run/8-worker threshold check:** the as-shipped BskSim MC campaign (2.269258 s) finished
  well under the 60 s threshold, so the full 100-run/8-worker one-orbit campaign was run directly —
  no 16-run extrapolation was needed.
- **Monte Carlo per-run "simulated seconds":** as in `RESULTS.md`, "total simulated seconds" in the
  table is runs × per-run duration (aggregate throughput across the whole campaign), not the
  per-run duration alone.

### `scenarioBskSimAttFeedbackMC_orbit.py` — edited copy details

`scripts/scenarioBskSimAttFeedbackMC_orbit.py` is a copy of
`basilisk-src/examples/MonteCarloExamples/scenarioBskSimAttFeedbackMC.py` with:
- `setExecutionCount(100)` and `setThreadCount(8)` (module-level `NUMBER_OF_RUNS`/`PROCESSES`
  constants, both set to 100/8 for this run).
- Its scenario dependency replaced with `scripts/scenario_AttFeedback_orbit.py` — a copy of
  `BskSim/scenarios/scenario_AttFeedback.py` with `runScenario()`'s stop time raised from 10 min to
  166 min (one orbit). Imported by its real on-disk module name
  (`import scenario_AttFeedback_orbit as scenario_AttFeedback`) with `scripts/` on `sys.path`,
  following the required real-module-name pattern so the MC controller's `multiprocessing.Pool`
  workers can pickle/re-import the scenario's module-level `scenario_AttFeedback`/`runScenario`
  references.
- Archive directory changed to `scenarioBskSimAttFeedbackMC_orbit` (under `scripts/`) to avoid
  colliding with the as-shipped run's output.
- Added `print(f"FAILURES={failures}")` plus the existing `assert len(failures) == 0` so the zero-
  failures claim below is drawn straight from the log, not inferred.

Log confirms (`logs/3x-31-scenarioBskSimAttFeedbackMC-orbit-100x8.log`):
```
FAILURES=[]
WALL_SECONDS=435.732685
PARENT_MAXRSS_KB_BEFORE=242280
PARENT_MAXRSS_KB_AFTER=243048
REAPED_CHILDREN_MAXRSS_KB=492840
```
**Zero runs failed** — `FAILURES=[]`, matching the passing `assert`.

## Peak memory — 100-run campaign

Recorded with the standard-library `resource` module in the parent driver process
(`scripts/run_mc_orbit.py`), around the `run(...)` call:
- Parent process peak RSS before the campaign: 242280 KB (~237 MB).
- Parent process peak RSS after the campaign: 243048 KB (~237 MB) — the parent itself barely grows;
  it dispatches work and waits on the pool.
- Peak RSS of the **worker (child)** processes, as seen by the parent via
  `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` (the largest single peak RSS among all reaped
  children, in KB): 492840 KB (~481 MB).

Caveat noted per instructions: worker processes are separate address spaces from the parent, so the
parent's own `getrusage(RUSAGE_SELF)` does not see their memory; `RUSAGE_CHILDREN` gives the single
largest per-child peak among reaped children, not a true aggregate/simultaneous total across all 8
concurrent workers. Getting a true wall-clock aggregate (sum of all 8 workers' RSS at one instant)
would need external sampling (e.g. polling `/proc/<pid>/status` for each worker PID while running)
rather than anything available directly to the parent process from the standard library after the
fact — considered beyond "easy"/standard-library scope per instructions, so not attempted.

## Warnings / errors verbatim

`grep -i "warn\|error\|exception\|traceback\|fail"` over every log in this follow-on
(`logs/3x-10-*.log` through `logs/3x-31-*.log`) found nothing except the expected, non-error
`FAILURES=[]` line in the 100-run campaign's log. No scenario raised a warning, error, exception,
or traceback; no run failed. The only other noteworthy console line (not an error) was the
Monte Carlo thread-count auto-adjustment already quoted above:
```
Fewer MCs spawned than processes assigned (4 < 8). Changing processes count to 4.
```

## Files

- Driver scripts: `scripts/run_feedbackrw3arg.py`, `scripts/run_module.py`,
  `scripts/run_mc_orbit.py`.
- Edited scenario copies (all in `scripts/`, per instructions — none of the sparse-checked-out
  `basilisk-src/` originals were modified):
  - `scripts/scenarioAttitudeFeedbackRW_orbit.py` — `scenarioAttitudeFeedbackRW.py` with
    `simulationTime` raised to `macros.min2nano(166.)`.
  - `scripts/scenario_FeedbackRW_orbit.py` — `BskSim/scenarios/scenario_FeedbackRW.py` with its
    relative `sys.path` setup replaced by absolute paths into the original
    `basilisk-src/examples/BskSim` tree (since this copy lives in a different directory), and the
    final `ConfigureStopTime` raised to `macros.min2nano(166.)`.
  - `scripts/scenario_AttFeedback_orbit.py` — `BskSim/scenarios/scenario_AttFeedback.py`, same
    `sys.path` fix, `runScenario()`'s stop time raised to `macros.min2nano(166.)`.
  - `scripts/scenarioBskSimAttFeedbackMC_orbit.py` — `MonteCarloExamples/scenarioBskSimAttFeedbackMC.py`,
    pointed at `scenario_AttFeedback_orbit.py`, run/thread counts raised to 100/8, as detailed above.
- Logs (3x- prefix, chronological): `logs/3x-10-scenarioAttitudeFeedbackRW-asshipped.log`,
  `logs/3x-11-scenarioAttitudeFeedbackRW-orbit.log`,
  `logs/3x-20-scenario_FeedbackRW-asshipped.log` (first run, includes one-time downloads),
  `logs/3x-20b-scenario_FeedbackRW-asshipped-cached.log` (representative re-run),
  `logs/3x-21-scenario_FeedbackRW-orbit.log`,
  `logs/3x-30-scenarioBskSimAttFeedbackMC-asshipped.log`,
  `logs/3x-31-scenarioBskSimAttFeedbackMC-orbit-100x8.log`.
