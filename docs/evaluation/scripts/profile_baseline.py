import sys, os, time, cProfile, pstats, io

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import scenario_FeedbackRW_orbit as scen

pr_construct = cProfile.Profile()
pr_execute = cProfile.Profile()
pr_postrun = cProfile.Profile()

# --- construction (module instantiation, SPICE kernel loading, gravity body setup) ---
t0 = time.perf_counter()
pr_construct.enable()
TheScenario = scen.scenario_AttitudeFeedbackRW()
pr_construct.disable()
t1 = time.perf_counter()

# --- InitializeSimulation + ExecuteSimulation (the C++ step loop behind the SWIG call) ---
pr_execute.enable()
TheScenario.InitializeSimulation()
t_init = time.perf_counter()

simulationTime = scen.macros.min2nano(10.)
TheScenario.modeRequest = 'inertial3D'
TheScenario.ConfigureStopTime(simulationTime)
TheScenario.ExecuteSimulation()
t_exec1 = time.perf_counter()

simulationTime = scen.macros.min2nano(166.)
TheScenario.modeRequest = 'directInertial3D'
TheScenario.ConfigureStopTime(simulationTime)
TheScenario.ExecuteSimulation()
pr_execute.disable()
t2 = time.perf_counter()

# --- post-run data retrieval and plotting ---
pr_postrun.enable()
figureList = TheScenario.pull_outputs(False)
pr_postrun.disable()
t3 = time.perf_counter()

print(f"CONSTRUCTION_SECONDS={t1 - t0:.6f}")
print(f"INIT_SECONDS={t_init - t1:.6f}")
print(f"EXECUTE1_SECONDS={t_exec1 - t_init:.6f}")
print(f"EXECUTE2_SECONDS={t2 - t_exec1:.6f}")
print(f"EXECUTE_TOTAL_SECONDS={(t2 - t1):.6f}")
print(f"POSTRUN_SECONDS={t3 - t2:.6f}")
print(f"TOTAL_SECONDS={t3 - t0:.6f}")

print("\n=== COMBINED TOP 25 BY CUMULATIVE TIME (construction + execute + postrun profiles merged) ===")
s = io.StringIO()
ps = pstats.Stats(pr_construct, pr_execute, pr_postrun, stream=s).sort_stats('cumulative')
ps.print_stats(25)
print(s.getvalue())

for label, pr in (("CONSTRUCTION", pr_construct), ("EXECUTE", pr_execute), ("POSTRUN", pr_postrun)):
    print(f"\n=== {label} PHASE: TOP 12 BY CUMULATIVE TIME (own profiler, not merged) ===")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(12)
    print(s.getvalue())
