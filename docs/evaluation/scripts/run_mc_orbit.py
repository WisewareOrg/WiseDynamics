import sys, os, time, importlib, resource

if __name__ == "__main__":
    path = sys.argv[1]
    n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else None
    n_workers = int(sys.argv[3]) if len(sys.argv) > 3 else None

    examples_dir = os.path.dirname(os.path.abspath(path))
    modname = os.path.splitext(os.path.basename(path))[0]
    sys.path.insert(0, examples_dir)
    mod = importlib.import_module(modname)

    if n_runs is not None:
        mod.NUMBER_OF_RUNS = n_runs
    if n_workers is not None:
        mod.PROCESSES = n_workers

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    mod.run(False)
    t1 = time.perf_counter()
    rss_after_self = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_children = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    print(f"WALL_SECONDS={t1 - t0:.6f}")
    print(f"PARENT_MAXRSS_KB_BEFORE={rss_before}")
    print(f"PARENT_MAXRSS_KB_AFTER={rss_after_self}")
    print(f"REAPED_CHILDREN_MAXRSS_KB={rss_children}")
