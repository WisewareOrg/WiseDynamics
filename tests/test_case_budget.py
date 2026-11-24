"""A case's native allocations per tick, layer by layer: each layer adds exactly its budget, by function.

A warm Leo500 case of the IMU spacecraft is built up one layer at a time, every layer flown through
`Scenario.run`:

1. `dynamics`: the vehicle's models and integrator, with no sensors, one recorded field and a flight
   software of Basilisk's own C++, so nothing crosses into Python;
2. `sensors`: the vehicle's sensor models at the flight-software period;
3. `recorders`: every field the vehicle records;
4. `host`: `FswHost` ticking a stub build that commands nothing, in place of the C++ flight software.

Each layer's budget is what it adds to the layer before it, as `native_allocations_by_function` charges
the allocations to functions, in two parts. Every function whose name is qualified with `::` (C++, such
as Basilisk's modules and WiseDynamics' integrator) has an exact figure of its own. Every other function
(C, such as the interpreter's, the C library's, and Basilisk's C modules) counts in one total, since how
those libraries' builds inline them moves an allocation between functions, not in or out of the total.
The counts are deterministic, so a budget is exact: an allocation added, removed or moved fails the layer
it lands in. They are those of the Basilisk and memray this repository locks and of the CPython it runs on:
an update to any of them can move a figure with no defect here, Basilisk's or CPython's by changing what
allocates, memray's by changing how it names a frame; the layer that fails names the functions. Every
layer's C total is zero, the host layer's included, since a run holds one thread state for its calls into
Python.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from tests.allocations import native_allocations_by_function
from tests.stubs import Guidance
from wise_dynamics.basilisk import FswHost, ImuSpacecraft
from wise_dynamics.basilisk.scenarios.leo500 import Leo500

TICKS = 200
"""The shorter flight's ticks; the counts of a deque's amortized blocks depend on it."""

LAYERS = ("dynamics", "sensors", "recorders", "host")

BUDGET: dict[str, dict[str, float]] = {
    "dynamics": {
        "HubEffector::computeHubOnlyDerivatives": 12.0,
        "Recorder<SCStatesMsgPayload>::UpdateState": 209 / TICKS,
        "HubEffector::matchGravitytoVelocityState": 1.0,
        "HubEffector::modifyStates": 1 / TICKS,
    },
    "sensors": {"Saturate::saturate": 2.0},
    "recorders": {"Recorder<IMUSensorMsgPayload>::UpdateState": 49 / TICKS},
    "host": {},
}
"""Each layer's native allocations per tick in C++ functions, beyond the layer before it, by function.

A figure over `TICKS` is a count of allocations a flight makes in blocks, such as a recorder's deque, whose
per-tick slope depends on the flight's length and on the payload's size."""


class _CommandingNothing:
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
    """The IMU spacecraft recording its attitude only, and with no sensors.

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
        """Declare the vehicle's attitude only, `truth/q_BN`.

        Returns:
            That field.
        """
        return {"truth/q_BN": super().recorded_fields()["truth/q_BN"]}


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
    fsw = FswHost(_CommandingNothing(), scenario.vehicle) if layer == "host" else Guidance()
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


def _added(per_layer: dict[str, dict[str, float]], layer: str) -> tuple[dict[str, float], dict[str, float]]:
    """What a layer allocates per tick beyond the layer before it: by C++ function, and by C function.

    A function the layer stops allocating in shows as a negative figure.

    Args:
        per_layer: Every layer's allocations per tick, by function.
        layer: The layer.

    Returns:
        The C++ functions' figures and the C functions' figures, nonzero ones only.
    """
    here = per_layer[layer]
    below = per_layer[LAYERS[LAYERS.index(layer) - 1]] if layer != LAYERS[0] else {}
    added = {function: here.get(function, 0.0) - below.get(function, 0.0) for function in here.keys() | below.keys()}
    cxx = {function: n for function, n in added.items() if "::" in function and n}
    c = {function: n for function, n in added.items() if "::" not in function and n}
    return cxx, c


@pytest.mark.parametrize("layer", LAYERS)
def test_each_layer_adds_exactly_its_allocation_budget(layer: str, per_layer: dict[str, dict[str, float]]):
    """The layer's C++ functions add exactly their budget, and its C functions add nothing.

    Args:
        layer: The layer under test.
        per_layer: Every layer's allocations per tick, by function.
    """
    cxx, c = _added(per_layer, layer)
    assert cxx == pytest.approx(BUDGET[layer], abs=1e-9), f"the {layer} layer's C++ adds {cxx}"
    assert sum(c.values()) == pytest.approx(0.0, abs=1e-9), f"the {layer} layer's C functions add {c} per tick"
