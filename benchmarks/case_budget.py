"""Time a warm Leo500 case layer by layer, and count its native allocations per tick by layer.

Each layer adds to the one before it, and the difference between two layers is that layer's cost:

1. `build`: the scenario and vehicle built and dispersed, nothing run.
2. `dynamics`: the vehicle's models and integrator, with no sensors, one recorded field and a
   flight software of Basilisk's own C++ that computes nothing the vehicle uses.
3. `sensors`: the vehicle's sensor models added at the flight-software period.
4. `recorders`: every field the vehicle records.
5. `host`: `FswHost` ticking a stub build that commands nothing, in place of the C++ flight software.
6. `run file`: the run file written.

Every layer flies through `Scenario.run` itself, so the layers measure the simulation a case flies;
the first two do so with a stand-in for the vehicle that has no sensors and records one field.
Every row times the whole case, from building the scenario to its run file. The layers are flown in
a shuffled order each round, from a seed the output names, and each row reports the median.

Run by hand from the repository root, never in CI:

    uv run python -m benchmarks.case_budget [--rounds 15] [--vehicle imu_spacecraft] [--allocations]

`--allocations` adds each layer's native allocations per tick and the functions that make them,
counted with memray from the dev group.
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
from Basilisk.fswAlgorithms import inertial3D

from wise_dynamics.basilisk import FswHost, Scenario, Vehicle
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


class CppFlightSoftware(inertial3D.inertial3D):
    """Basilisk's own C++ `inertial3D` in the flight-software slot, so no call crosses into Python.

    Attributes:
        period_s: Its period, in seconds.
    """

    period_s = PERIOD_S


def stripped(vehicle: type[Vehicle], *, sensors: bool) -> type[Vehicle]:
    """A stand-in for `vehicle` recording only its first field and returning it unfinished.

    Args:
        vehicle: The vehicle class.
        sensors: Whether the stand-in keeps its sensors.

    Returns:
        The stand-in class.
    """

    class Stripped(vehicle):  # ty: ignore[unsupported-base]
        """The vehicle with its first recorded field only, `finish` left as `Vehicle`'s."""

        def sensor_models(self, sample_period_s: float) -> list[Any]:
            """List the sensors, or none.

            Args:
                sample_period_s: The period the sensors sample at, in seconds.

            Returns:
                The vehicle's sensors when the stand-in keeps them, else none.
            """
            return super().sensor_models(sample_period_s) if sensors else []

        def recorded_fields(self) -> dict[str, tuple[Any, str, str]]:
            """Declare the vehicle's first recorded field only.

            Returns:
                That field.
            """
            first = next(iter(super().recorded_fields().items()))
            return dict([first])

        def finish(self, datasets: Mapping[str, tuple[Any, str]]) -> dict[str, tuple[Any, str]]:
            """Return the datasets as recorded.

            Args:
                datasets: Dataset name to `(array, units)`.

            Returns:
                The same datasets.
            """
            return dict(datasets)

    return Stripped


def fly(depth: int, vehicle: str, case: int, directory: Path, ticks: int = round(DURATION_S / PERIOD_S)) -> None:
    """Fly one case built up to the layer at index `depth` of `LAYERS`.

    Args:
        depth: The index in `LAYERS` of the last layer included.
        vehicle: The vehicle's name in `VEHICLES`.
        case: The case number.
        directory: Where the run file is written and removed.
        ticks: The number of flight-software periods to fly.
    """
    vehicle_type = VEHICLES[vehicle]
    if depth <= LAYERS.index("sensors"):
        vehicle_type = stripped(vehicle_type, sensors=depth == LAYERS.index("sensors"))
    scenario: Scenario = SCENARIOS["leo500"](vehicle_type())
    scenario.disperse_case(np.random.SeedSequence(0, spawn_key=(case,)))
    if depth == LAYERS.index("build"):
        return
    craft = scenario.vehicle
    fsw = FswHost(StubBuild(), craft) if depth >= LAYERS.index("host") else CppFlightSoftware()
    datasets = scenario.run(fsw, {}, duration_s=ticks * PERIOD_S)
    if depth >= LAYERS.index("run file"):
        run_file = directory / f"run_{case}.h5"
        craft.write_run_file(run_file, datasets, {"case": case})
        run_file.unlink()


def time_layers(vehicle: str, rounds: int, seed: int, directory: Path) -> dict[str, float]:
    """Median wall time of each layer over `rounds` rounds, the layers shuffled every round, after one warm-up.

    Args:
        vehicle: The vehicle's name in `VEHICLES`.
        rounds: The number of rounds.
        seed: The seed of the shuffles.
        directory: Where run files are written and removed.

    Returns:
        Layer name to its median, in milliseconds.
    """
    shuffler = random.Random(seed)
    for depth in range(len(LAYERS)):
        fly(depth, vehicle, 0, directory)
    times: dict[str, list[float]] = {name: [] for name in LAYERS}
    for case in range(rounds):
        order = list(range(len(LAYERS)))
        shuffler.shuffle(order)
        for depth in order:
            gc.collect()
            start = time.perf_counter()
            fly(depth, vehicle, case, directory)
            times[LAYERS[depth]].append(time.perf_counter() - start)
    return {name: statistics.median(t) * 1e3 for name, t in times.items()}


def allocations(vehicle: str, directory: Path) -> dict[str, dict[str, float]]:
    """Each run layer's native allocations per tick, by the function that makes them.

    Args:
        vehicle: The vehicle's name in `VEHICLES`.
        directory: Where memray's capture files are written and removed.

    Returns:
        Layer name to function name to allocations per tick, for the layers that run a simulation.
    """
    from tests.allocations import native_allocations_by_function

    def flight(depth: int) -> Callable[[int], None]:
        """Fly cases built up to one layer.

        Args:
            depth: The index in `LAYERS` of the last layer included.

        Returns:
            A function flying the given number of ticks.
        """
        return lambda ticks: fly(depth, vehicle, 0, directory, ticks=ticks)

    run_layers = range(LAYERS.index("dynamics"), LAYERS.index("run file"))
    return {LAYERS[d]: native_allocations_by_function(flight(d), ticks=200, directory=directory) for d in run_layers}


def main() -> None:
    """Print each layer's median, its difference from the layer before, and its allocations per tick."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--vehicle", default="imu_spacecraft", choices=sorted(VEHICLES))
    parser.add_argument("--allocations", action="store_true")
    parser.add_argument("--seed", type=int, default=random.SystemRandom().randrange(2**32))
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp())
    medians = time_layers(args.vehicle, args.rounds, args.seed, directory)
    by_layer = allocations(args.vehicle, directory) if args.allocations else {}
    print(
        f"Leo500, {args.vehicle}, {DURATION_S:.0f} s at {PERIOD_S} s; median of {args.rounds} rounds, seed {args.seed}"
    )
    print(f"{'layer':10s} {'ms':>8s} {'+ms':>8s} {'allocs/tick':>12s}")
    before_ms = 0.0
    before: dict[str, float] = {}
    for name in LAYERS:
        functions = by_layer.get(name)
        total = f"{sum(functions.values()):12.1f}" if functions is not None else f"{'':12s}"
        print(f"{name:10s} {medians[name]:8.1f} {medians[name] - before_ms:8.1f} {total}")
        for function, per_tick in (functions or {}).items():
            added = per_tick - before.get(function, 0.0)
            if added:
                print(f"{'':30s} {added:+8.2f}  {function}")
        before_ms = medians[name]
        before = functions if functions is not None else before


if __name__ == "__main__":
    main()
