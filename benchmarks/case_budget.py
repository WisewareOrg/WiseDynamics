"""Time a warm Leo500 case layer by layer, and count its native allocations per tick by layer.

Each layer adds to the one before it, and the difference between two layers is that layer's cost:

1. `build`: the scenario and vehicle built and dispersed, nothing run.
2. `dynamics`: the vehicle's models and integrator, stepped with no sensors, recorders or host.
3. `sensors`: the vehicle's sensor models added at the flight-software period.
4. `recorders`: one recorder per recorded field.
5. `host`: `FswHost` ticking a stub build that commands nothing, at the flight-software period.
6. `run file`: the recorders read, the vehicle's `finish`, and the run file written.

Layers 2 to 5 lay out the simulation as `Scenario.run` does; the last row, `Scenario.run`, flies the
case through `Scenario.run` itself and writes its run file, so the layered total can be checked
against the real path. Every row times the whole case, from building the scenario to tearing the
simulation down. The layers are flown in a shuffled order each round, and each row reports the
median over the rounds.

Run by hand, never in CI:

    uv run python benchmarks/case_budget.py [--rounds 15] [--vehicle imu_spacecraft] [--allocations]

`--allocations` counts native allocations per tick for layers 2 to 5 with memray, from the dev group.
"""

import argparse
import gc
import random
import statistics
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

from wise_dynamics.basilisk import FswHost, Scenario
from wise_dynamics.basilisk.scenario import _FIDELITY, _step_ns
from wise_dynamics.basilisk.scenarios import SCENARIOS
from wise_dynamics.basilisk.vehicles import VEHICLES

DURATION_S = 5700.0
PERIOD_S = 1.0
LAYERS = ("build", "dynamics", "sensors", "recorders", "host", "run file")


class StubBuild:
    """A flight-software build that commands nothing.

    Attributes:
        period_s: The build's period, in seconds.
    """

    period_s = PERIOD_S

    def boot(self) -> None:
        """Start the build: nothing to start."""

    def tick(self, t_ns: int, messages_by_role: Mapping[str, Any]) -> Mapping[str, Any]:
        """Command nothing.

        Args:
            t_ns: The simulation time, in nanoseconds.
            messages_by_role: The vehicle's readings by role.

        Returns:
            No commands.
        """
        return {}


def build(vehicle: str, case: int) -> Scenario:
    """Build Leo500 flying a new vehicle, dispersed as the plugin disperses case `case`.

    Args:
        vehicle: The vehicle's name in `VEHICLES`.
        case: The case number.

    Returns:
        The dispersed scenario.
    """
    scenario = SCENARIOS["leo500"](VEHICLES[vehicle]())
    scenario.disperse_case(np.random.SeedSequence(0, spawn_key=(case,)))
    return scenario


def fly_layer(depth: int, vehicle: str, case: int, directory: Path, duration_s: float = DURATION_S) -> None:
    """Fly one case built up to the layer at index `depth` of `LAYERS`.

    Args:
        depth: The index in `LAYERS` of the last layer included.
        vehicle: The vehicle's name in `VEHICLES`.
        case: The case number.
        directory: Where the run file is written and removed.
        duration_s: Simulated time, in seconds.
    """
    scenario = build(vehicle, case)
    if depth == 0:
        return
    craft = scenario.vehicle
    simulation = SimulationBaseClass.SimBaseClass()
    fsw_ns = macros.sec2nano(PERIOD_S)
    step_ns = _step_ns(fsw_ns)
    craft.spacecraft.setIntegrator(_FIDELITY[scenario.fidelity](craft.spacecraft))
    process = simulation.CreateNewProcess("scenario_process", 2)
    for name, task_ns, priority in (
        ("dynamics", step_ns, 3),
        ("sensor", fsw_ns, 2),
        ("fsw", fsw_ns, 1),
        ("record", step_ns, 0),
    ):
        process.addTask(simulation.CreateNewTask(name, task_ns), priority)
    for model in craft.models():
        simulation.AddModelToTask("dynamics", model)
    if depth >= 2:
        for sensor in craft.sensor_models(PERIOD_S):
            simulation.AddModelToTask("sensor", sensor)
    fields = craft.recorded_fields()
    recorders = {}
    if depth >= 3:
        recorders = {path: message.recorder(step_ns) for path, (message, _, _) in fields.items()}
        for recorder in recorders.values():
            simulation.AddModelToTask("record", recorder)
    if depth >= 4:
        simulation.AddModelToTask("fsw", FswHost(StubBuild(), craft))
    simulation.InitializeSimulation()
    simulation.ConfigureStopTime(macros.sec2nano(duration_s))
    simulation.ExecuteSimulation()
    if depth >= 5:
        first = next(iter(recorders.values()))
        datasets = {"t_s": (np.asarray(first.times()) / macros.sec2nano(1.0), "s")}
        for path, (_, field, units) in fields.items():
            datasets[path] = (np.array(getattr(recorders[path], field)), units)
        run_file = directory / f"layer_{case}.h5"
        craft.write_run_file(run_file, craft.finish(datasets), {"case": case})
        run_file.unlink()
    del simulation


def fly_real(vehicle: str, case: int, directory: Path) -> None:
    """Fly one case through `Scenario.run` with the stub build, and write and remove its run file.

    Args:
        vehicle: The vehicle's name in `VEHICLES`.
        case: The case number.
        directory: Where the run file is written and removed.
    """
    scenario = build(vehicle, case)
    datasets = scenario.run(FswHost(StubBuild(), scenario.vehicle), {}, duration_s=DURATION_S)
    run_file = directory / f"real_{case}.h5"
    scenario.vehicle.write_run_file(run_file, datasets, {"case": case})
    run_file.unlink()


def time_rows(rows: Mapping[str, Callable[[int], None]], rounds: int) -> dict[str, float]:
    """Median wall time of each row over `rounds` rounds, the rows shuffled every round, after one warm-up.

    Args:
        rows: Row name to a function flying one case of the given number.
        rounds: The number of rounds.

    Returns:
        Row name to its median, in milliseconds.
    """
    for fly in rows.values():
        fly(0)
    times: dict[str, list[float]] = {name: [] for name in rows}
    for case in range(rounds):
        order = list(rows)
        random.shuffle(order)
        for name in order:
            gc.collect()
            start = time.perf_counter()
            rows[name](case)
            times[name].append(time.perf_counter() - start)
    return {name: statistics.median(t) * 1e3 for name, t in times.items()}


def allocations_per_tick(vehicle: str, directory: Path) -> dict[str, float]:
    """Native allocations per tick of layers 2 to 5, the slope between a 200 and a 400 tick flight.

    Args:
        vehicle: The vehicle's name in `VEHICLES`.
        directory: Where memray's capture files are written and removed.

    Returns:
        Layer name to its allocations per tick, the layers before it included.
    """
    import memray
    from memray._memray import AllocatorType

    releases = {AllocatorType.FREE, AllocatorType.MUNMAP, AllocatorType.PYMALLOC_FREE}

    def count(depth: int, ticks: int) -> int:
        """Native allocations of one flight of `ticks` ticks.

        Args:
            depth: The index in `LAYERS` of the last layer included.
            ticks: The number of ticks to fly.

        Returns:
            The number of allocations.
        """
        capture = directory / "capture.bin"
        with memray.Tracker(capture, trace_python_allocators=False):
            fly_layer(depth, vehicle, 0, directory, duration_s=ticks * PERIOD_S)
        records = memray.FileReader(capture).get_allocation_records()
        total = sum(r.n_allocations for r in records if AllocatorType(r.allocator) not in releases)
        capture.unlink()
        return total

    return {LAYERS[depth]: (count(depth, 400) - count(depth, 200)) / 200 for depth in range(1, 5)}


def main() -> None:
    """Print the budget: each row's median, its difference from the row before, and its allocations per tick."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--vehicle", default="imu_spacecraft", choices=sorted(VEHICLES))
    parser.add_argument("--allocations", action="store_true")
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp())

    def layer(depth: int) -> Callable[[int], None]:
        """Fly cases built up to one layer.

        Args:
            depth: The index in `LAYERS` of the last layer included.

        Returns:
            A function flying the case of the given number.
        """
        return lambda case: fly_layer(depth, args.vehicle, case, directory)

    rows = {name: layer(depth) for depth, name in enumerate(LAYERS)}
    rows["Scenario.run"] = lambda case: fly_real(args.vehicle, case, directory)
    medians = time_rows(rows, args.rounds)
    allocations = allocations_per_tick(args.vehicle, directory) if args.allocations else {}
    print(f"Leo500, {args.vehicle}, {DURATION_S:.0f} s at {PERIOD_S} s, median of {args.rounds} rounds")
    print(f"{'row':14s} {'ms':>8s} {'+ms':>8s} {'allocs/tick':>12s}")
    before = 0.0
    for name in LAYERS:
        per_tick = f"{allocations[name]:12.1f}" if name in allocations else f"{'':12s}"
        print(f"{name:14s} {medians[name]:8.1f} {medians[name] - before:8.1f} {per_tick}")
        before = medians[name]
    print(f"{'Scenario.run':14s} {medians['Scenario.run']:8.1f}")


if __name__ == "__main__":
    main()
