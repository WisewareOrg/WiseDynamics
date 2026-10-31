# Basilisk (`bsk`) eval results

- **bsk version:** 2.12.1
- **Python version:** 3.13.15 (venv at `/home/wise/code/bsk-eval/.venv`, built with `uv venv --python 3.13`)
- **Install command:** `uv pip install --python .venv/bin/python "bsk[all]"`
- **Install wall time:** 21.145 s (`uv` reported `6.40s user 1.82s system 38% cpu 21.145 total`; see `logs/01-install.log`)
- **venv creation wall time:** 0.026 s (see `logs/00-venv-create.log`)
- **Import check:** `import Basilisk` and `from Basilisk.simulation import spacecraft` both succeeded with no errors (see `logs/02-import-check.log`). No loader/shared-library problems — nix-ld worked as expected.
- **Examples source:** sparse, shallow clone of `https://github.com/AVSLab/basilisk` at tag `v2.12.1` into `basilisk-src/`, checked out to `examples/` only via `git sparse-checkout set examples`. Not built from source — only used for the example scripts.

## Runs

| Scenario | Runs | Workers | Simulated seconds | Wall seconds | Ratio (sim/wall) |
|---|---|---|---|---|---|
| scenarioAttitudePointing.py | 1 | 1 | 600 | 0.189391 | 3168.0 |
| scenarioAttitudePointingPy.py | 1 | 1 | 600 | 0.461949 | 1298.8 |
| scenarioMonteCarloAttRW.py (as shipped) | 4 | 4 (default `mp.cpu_count()`=8, capped to 4 since fewer MC runs than processes) | 2400 | 1.619820 | 1481.6 |
| scenarioMonteCarloAttRW.py (100 runs, 8 workers) | 100 | 8 | 60000 | 20.818339 | 2882.1 |

Notes on the table:
- "Simulated seconds" for the single-run scenarios is the `scSim.ConfigureStopTime(simulationTime)` value in each script: `simulationTime = macros.min2nano(10.)` = 600 s, for both `scenarioAttitudePointing.py` and `scenarioAttitudePointingPy.py`.
- `scenarioMonteCarloAttRW.py` dispatches multiple independent Monte Carlo runs, each simulating the same 600 s (`simulationTime = macros.min2nano(10.)` inside `createScenarioAttitudeFeedbackRW`). "Simulated seconds" in the table is the **total** across all runs (runs × 600 s) so the ratio reflects aggregate simulated-work throughput for the whole campaign; this is a choice made to get a measurement — the ambiguity is that "simulated seconds" could also be read as the per-run 600 s, in which case the ratio would instead describe an effective clock-for-clock speedup and be far higher.
- Each timing is wall time of the `run(...)` call only (`time.perf_counter()` around it, via small driver scripts in `scripts/`), not including Python interpreter startup or Basilisk module import. `/usr/bin/time` is not present on this machine (NixOS), so `time.perf_counter()` was used throughout instead of `/usr/bin/time -v`.
- All runs executed with `MPLBACKEND=Agg` and `show_plots=False` (or the `show_plots` argument omitted/False where the signature has no second positional parameter).

### Monte Carlo campaign — as shipped

- `scenarioMonteCarloAttRW.py` defines `NUMBER_OF_RUNS = 4` at module scope (line 247) and calls `monteCarlo.setExecutionCount(NUMBER_OF_RUNS)`.
- The number of worker processes is set via `monteCarlo.setThreadCount(...)`, but that call is commented out in the shipped script (`# monteCarlo.setThreadCount(PROCESSES)`, and `PROCESSES` is not even defined anywhere in the file). With no explicit call, the `Controller` defaults `self.numProcess = mp.cpu_count()` (8 on this machine). The log confirms: `Beginning simulation with 4 runs on 8 threads` followed immediately by `Fewer MCs spawned than processes assigned (4 < 8). Changing processes count to 4.` — so the effective worker count for the as-shipped run is 4.
- Both the run count and the worker count are configurable (`NUMBER_OF_RUNS` constant; `monteCarlo.setThreadCount(n)`), just not wired to a CLI flag.
- As-shipped run finished in 1.62 s, well under the 2-minute threshold, so the 100-run/8-worker variant was also run per instructions.

### Monte Carlo campaign — 100 runs, 8 workers

- Ran from a copy of the script, `basilisk-src/examples/scenarioMonteCarloAttRW_100x8.py`, with `NUMBER_OF_RUNS = 100` and `monteCarlo.setThreadCount(8)` uncommented/set (everything else unchanged). Editing was done on the sparse-checked-out copy inside `bsk-eval/basilisk-src`, not on anything outside this directory.
- Completed in 20.82 s with all 100 runs succeeding (log shows `Beginning simulation with 100 runs on 8 threads` and ends with `Thread ... Job 99 finished successfully`).

## Warnings / errors verbatim

No warnings or errors were emitted in any of the four successful runs (`logs/10-*.log`, `logs/11-*.log`, `logs/20-*.log`, `logs/21-*.log`); `grep -i "warn\|error\|exception\|traceback"` over each returns nothing.

One failure occurred during setup and was corrected before the recorded runs above — kept here for completeness since it is a real finding about the environment, not a scenario failure:

```
Unknown exception while trying to pickle monte-carlo-controller... 
continuing...

 Can't pickle <function createScenarioAttitudeFeedbackRW at 0x744483eae340>: import of module 'scenario_mod' failed
...
_pickle.PicklingError: Can't pickle <function createScenarioAttitudeFeedbackRW at 0x744483eae340>: import of module 'scenario_mod' failed
...
Unknown exception while running simulations: Can't pickle <function createScenarioAttitudeFeedbackRW at 0x744483eae340>: import of module 'scenario_mod' failed
Failed [0, 1, 2, 3] saving to 'failures.txt'
Traceback (most recent call last):
  ...
  File "/home/wise/code/bsk-eval/basilisk-src/examples/scenarioMonteCarloAttRW.py", line 415, in run
    assert len(failures) == 0, "No runs should fail"
AssertionError: No runs should fail
```

This happened because the first version of the driver script loaded the scenario module via `importlib.util.spec_from_file_location` under a synthetic module name (`scenario_mod`) not registered on `sys.path`. `MonteCarlo.Controller` uses `multiprocessing.Pool` and needs to pickle the scenario's model-construction function by module+qualname reference; a synthetic, non-importable module name breaks that. Fixed by adding the real `examples/` directory to `sys.path` and importing the script by its real filename-derived module name (`importlib.import_module("scenarioMonteCarloAttRW")`) instead, which made the function picklable. This only affected the Monte Carlo scenario (the only one using multiprocessing) — the two single-run scenarios worked with the original approach.

## Non-obvious steps

- `/usr/bin/time` does not exist on this NixOS host; used Python `time.perf_counter()` wrapped around each `run(...)` call instead.
- `git clone --branch v2.12.1` on this repo prints `warning: refs/tags/v2.12.1 ... is not a commit!` — harmless; it's an annotated-tag dereference warning from the shallow/filtered clone, and `git sparse-checkout` + `git checkout v2.12.1` still resolved and checked out the correct tagged tree.
- The Monte Carlo scenario's worker/process count is not exposed as a script argument; raising it required editing a commented-out line in a copy of the script (`scenarioMonteCarloAttRW_100x8.py`), not a CLI flag.
- Multiprocessing-based scenarios (the Monte Carlo one) require the scenario module to be import-reachable by its real module name on `sys.path` for pickling to work when driven from an external script; loading it as an anonymous module via `importlib.util` fails inside worker processes.
