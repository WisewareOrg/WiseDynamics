import sys, os, time

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)

import scenario_FeedbackRW_orbit as scen

t0 = time.perf_counter()
TheScenario = scen.scenario_AttitudeFeedbackRW()
t1 = time.perf_counter()

TheScenario.InitializeSimulation()
t2 = time.perf_counter()

simulationTime = scen.macros.min2nano(10.)
TheScenario.modeRequest = 'inertial3D'
TheScenario.ConfigureStopTime(simulationTime)
TheScenario.ExecuteSimulation()
t3 = time.perf_counter()

simulationTime = scen.macros.min2nano(166.)
TheScenario.modeRequest = 'directInertial3D'
TheScenario.ConfigureStopTime(simulationTime)
TheScenario.ExecuteSimulation()
t4 = time.perf_counter()

figureList = TheScenario.pull_outputs(False)
t5 = time.perf_counter()

print(f"CONSTRUCTION_SECONDS={t1 - t0:.6f}")
print(f"INIT_SECONDS={t2 - t1:.6f}")
print(f"EXECUTE1_SECONDS={t3 - t2:.6f}")
print(f"EXECUTE2_SECONDS={t4 - t3:.6f}")
print(f"EXECUTE_TOTAL_SECONDS={(t4 - t2):.6f}")
print(f"POSTRUN_SECONDS={t5 - t4:.6f}")
print(f"TOTAL_SECONDS={t5 - t0:.6f}")

# attitude error scalar at end of run: norm of final recorded sigma_BR row
import numpy as np
sigma_BR_final = TheScenario.attErrRec.sigma_BR[-1]
print(f"FINAL_SIGMA_BR_NORM={np.linalg.norm(sigma_BR_final):.9e}")
