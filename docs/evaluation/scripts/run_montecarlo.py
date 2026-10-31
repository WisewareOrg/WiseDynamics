import sys, os, time, importlib

if __name__ == "__main__":
    path = sys.argv[1]
    examples_dir = os.path.dirname(os.path.abspath(path))
    modname = os.path.splitext(os.path.basename(path))[0]
    sys.path.insert(0, examples_dir)
    mod = importlib.import_module(modname)

    t0 = time.perf_counter()
    dirName = mod.run(saveFigures=False, case=1, show_plots=False, delete_data=True, useBokeh=False)
    t1 = time.perf_counter()
    print(f"WALL_SECONDS={t1 - t0:.6f}")
    print(f"DIRNAME={dirName}")
