import sys, os, time, importlib
import numpy as np

if __name__ == "__main__":
    path = sys.argv[1]
    scripts_dir = os.path.dirname(os.path.abspath(path))
    modname = os.path.splitext(os.path.basename(path))[0]
    sys.path.insert(0, scripts_dir)
    mod = importlib.import_module(modname)

    t0 = time.perf_counter()
    TheScenario = mod.scenario_AttitudeFeedbackRW()
    t1 = time.perf_counter()

    mod.runScenario(TheScenario)
    t2 = time.perf_counter()

    figureList = TheScenario.pull_outputs(False)
    t3 = time.perf_counter()

    final_time_s = TheScenario.attErrRec.times()[-1] / 1.0e9
    sigma_BR_final = TheScenario.attErrRec.sigma_BR[-1]
    final_sigma_norm = float(np.linalg.norm(sigma_BR_final))

    print(f"CONSTRUCTION_SECONDS={t1 - t0:.6f}")
    print(f"RUNSCENARIO_SECONDS={t2 - t1:.6f}")
    print(f"POSTRUN_SECONDS={t3 - t2:.6f}")
    print(f"TOTAL_SECONDS={t3 - t0:.6f}")
    print(f"FINAL_SIM_TIME_S={final_time_s:.3f}")
    print(f"FINAL_SIGMA_BR_NORM={final_sigma_norm:.9e}")
