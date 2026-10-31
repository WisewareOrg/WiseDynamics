import sys, time, importlib.util
path = sys.argv[1]
spec = importlib.util.spec_from_file_location("scenario_mod", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

t0 = time.perf_counter()
mod.run(False)
t1 = time.perf_counter()
print(f"WALL_SECONDS={t1 - t0:.6f}")
