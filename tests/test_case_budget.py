"""A case's native allocations per tick, layer by layer: each layer adds exactly its budget, by function.

A warm Leo500 case of the IMU spacecraft is built up one layer at a time, every layer flown through
`Scenario.run`:

1. `dynamics`: the vehicle's models and integrator, with no sensors, one recorded field and a flight
   software of Basilisk's own C++, so nothing crosses into Python;
2. `sensors`: the vehicle's sensor models at the flight-software period;
3. `recorders`: every field the vehicle records;
4. `host`: `FswHost` ticking a stub build that commands nothing, in place of the C++ flight software.

Each layer's budget is what it adds to the layer before it, by the function that makes each allocation,
as `native_allocations_by_function` charges them. The counts are deterministic, so a budget is an
exact figure: an allocation added, removed or moved fails the layer it lands in. The counts are of the
bsk this repository locks; Python's own allocations depend on its minor version.
"""

import sys
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from Basilisk.fswAlgorithms import inertial3D

from tests.allocations import native_allocations_by_function
from wise_dynamics.basilisk import FswHost, ImuSpacecraft, Vehicle
from wise_dynamics.basilisk.scenarios.leo500 import Leo500

TICKS = 200
"""The shorter flight's ticks; the counts of a deque's amortized blocks depend on it."""

LAYERS = ("dynamics", "sensors", "recorders", "host")

_THREAD_STATE_PER_CALL = {
    (3, 12): {"new_threadstate": 1.0, "_PyMem_ArenaAlloc": 1.0},
    (3, 14): {
        "new_threadstate": 1.0,
        "_PyMem_ArenaAlloc": 1.0,
        "pthread_getattr_np": 1.0,
        "__pthread_attr_extension": 1.0,
        "pthread_attr_setaffinity_np": 1.0,
    },
}
"""What CPython allocates as Basilisk's thread calls into Python with no thread state, by version."""

BUDGET: dict[str, dict[str, float]] = {
    "dynamics": {
        "ExtendedStateVector::fromStateData": 91.0,
        "ExtendedStateVector::map": 65.0,
        "svIntegratorRungeKutta<3ul>::propagateStateWithKVectors": 13.0,
        "HubEffector::computeHubOnlyDerivatives": 12.0,
        "Recorder<SCStatesMsgPayload>::UpdateState": 1.045,
        "HubEffector::matchGravitytoVelocityState": 1.0,
        "HubEffector::modifyStates": 0.005,
    },
    "sensors": {"Saturate::saturate": 2.0},
    "recorders": {
        "Recorder<SCStatesMsgPayload>::UpdateState": 3.135,
        "Recorder<IMUSensorMsgPayload>::UpdateState": 0.49,
    },
    "host": _THREAD_STATE_PER_CALL.get(sys.version_info[:2], {}),
}
"""Each layer's native allocations per tick, beyond the layer before it, by the function that makes them."""


class _CppFlightSoftware(inertial3D.inertial3D):
    """Basilisk's own C++ `inertial3D` in the flight-software slot, so no call crosses into Python.

    Attributes:
        period_s: Its period, in seconds.
    """

    period_s = 1.0


class _Commanding:
    """A flight-software build that commands nothing.

    Attributes:
        period_s: The build's period, in seconds.
    """

    period_s = 1.0

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


class _Stripped(ImuSpacecraft):
    """The IMU spacecraft recording its first field only, leaving it unfinished, and with no sensors.

    Attributes:
        sensing: Whether the stand-in keeps its sensors.
    """

    sensing = False

    def sensor_models(self, sample_period_s: float) -> list[Any]:
        """List the sensors, or none.

        Args:
            sample_period_s: The period the sensors sample at, in seconds.

        Returns:
            The vehicle's sensors when the stand-in keeps them, else none.
        """
        return super().sensor_models(sample_period_s) if self.sensing else []

    def recorded_fields(self) -> dict[str, tuple[Any, str, str]]:
        """Declare the vehicle's first recorded field only.

        Returns:
            That field.
        """
        return dict([next(iter(super().recorded_fields().items()))])

    def finish(self, datasets: Mapping[str, tuple[Any, str]]) -> dict[str, tuple[Any, str]]:
        """Return the datasets as recorded.

        Args:
            datasets: Dataset name to `(array, units)`.

        Returns:
            The same datasets.
        """
        return Vehicle.finish(self, datasets)


class _Sensing(_Stripped):
    """The stand-in with its sensors.

    Attributes:
        sensing: Whether the stand-in keeps its sensors.
    """

    sensing = True


_VEHICLE = {"dynamics": _Stripped, "sensors": _Sensing, "recorders": ImuSpacecraft, "host": ImuSpacecraft}


def _fly(layer: str, ticks: int) -> None:
    """Fly one warm Leo500 case built up to `layer`.

    Args:
        layer: The last layer included.
        ticks: The number of one-second ticks to fly.
    """
    scenario = Leo500(_VEHICLE[layer]())
    scenario.disperse_case(np.random.SeedSequence(0, spawn_key=(0,)))
    fsw = FswHost(_Commanding(), scenario.vehicle) if layer == "host" else _CppFlightSoftware()
    scenario.run(fsw, {}, duration_s=float(ticks))


@pytest.fixture(scope="module")
def per_layer(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, float]]:
    """Every layer's native allocations per tick, by function, the layers before it included.

    Args:
        tmp_path_factory: Makes the directory memray's captures are written to.

    Returns:
        Layer name to function name to allocations per tick.
    """
    directory = tmp_path_factory.mktemp("captures")
    return {
        layer: native_allocations_by_function(
            lambda ticks, layer=layer: _fly(layer, ticks), ticks=TICKS, directory=directory
        )
        for layer in LAYERS
    }


@pytest.mark.parametrize("layer", LAYERS)
def test_each_layer_adds_exactly_its_allocation_budget(layer: str, per_layer: dict[str, dict[str, float]]):
    """What the layer allocates per tick beyond the layer before it is its budget, function by function.

    Args:
        layer: The layer under test.
        per_layer: Every layer's allocations per tick, by function.
    """
    below = per_layer[LAYERS[LAYERS.index(layer) - 1]] if layer != LAYERS[0] else {}
    added = {
        function: per_tick - below.get(function, 0.0)
        for function, per_tick in per_layer[layer].items()
        if per_tick != below.get(function, 0.0)
    }
    assert added == pytest.approx(BUDGET[layer], abs=1e-9), f"the {layer} layer adds {added}"
