"""The compiled Runge-Kutta: bit-identical to Basilisk's, allocating nothing per step, refusing what it cannot run."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import Basilisk
import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.simulation import spacecraft as bsk_spacecraft
from Basilisk.simulation import svIntegrators

from tests.allocations import assert_no_native_allocations_per_tick
from tests.stubs import Guidance
from wise_dynamics.basilisk import FswHost, ImuSpacecraft, scenario
from wise_dynamics.basilisk import _runge_kutta as runge_kutta
from wise_dynamics.basilisk.scenarios import SCENARIOS
from wise_dynamics.basilisk.scenarios.leo500 import Leo500
from wise_dynamics.basilisk.vehicles import VEHICLES

_BASILISK_ALLOCATING_IN_A_STEP = ["HubEffector::computeHubOnlyDerivatives", "Spacecraft::equationsOfMotion"]
"""The Basilisk functions an integrator step calls that allocate per step, by Basilisk's own doing.

The equations of motion are named where a vehicle has effectors; with the hub alone, the optimised
build inlines them, and the hub's derivatives are the innermost Basilisk frame."""

_BUILD_FACTS = [
    "cxx.layoutCanaries.stdStringSize",
    "cxx.layoutCanaries.stdVectorIntSize",
    "cxx.layoutCanaries.sysModelAlignment",
    "cxx.layoutCanaries.sysModelSize",
    "cxx.standardLibrary.useCxx11Abi",
    "dependencies.eigen.matrix3dSize",
    "dependencies.eigen.maxAlignBytes",
    "dependencies.eigen.maxStaticAlignBytes",
    "dependencies.eigen.vectorXdSize",
    "dependencies.eigen.vectorization",
    "dependencies.eigen.version",
    "target.architecture",
    "target.instructionSet",
]
"""The build facts the library must report, by their keys in `Basilisk.getBuildInfo()["abi"]`."""


def _basilisk_rk3(craft: Any) -> Any:
    """Give a spacecraft Basilisk's own three-stage Runge-Kutta with Ralston's tableau, written out here.

    The tableau is the literal `lo` integrated with before the compiled integrator, not the one
    `scenario` holds, so a change to that one fails the comparison.

    Args:
        craft: The Basilisk spacecraft.

    Returns:
        The integrator.
    """
    integrator = svIntegrators.svIntegratorRungeKutta3(
        craft,
        [[0.0, 0.0, 0.0], [1 / 2, 0.0, 0.0], [0.0, 3 / 4, 0.0]],
        [2 / 9, 1 / 3, 4 / 9],
        [0.0, 1 / 2, 3 / 4],
    )
    craft.setIntegrator(integrator)
    return integrator


def _basilisk_rk4(craft: Any) -> Any:
    """Give a spacecraft Basilisk's own classical fourth-order Runge-Kutta, whose tableau it holds itself.

    Args:
        craft: The Basilisk spacecraft.

    Returns:
        The integrator.
    """
    integrator = svIntegrators.svIntegratorRK4(craft)
    craft.setIntegrator(integrator)
    return integrator


_BASILISK = {"lo": _basilisk_rk3, "med": _basilisk_rk4}


def _assert_bit_identical(compiled: dict, basilisk: dict) -> None:
    """Hold two runs' datasets to the same names, units, dtypes, shapes and bytes.

    Args:
        compiled: The run with the compiled integrator.
        basilisk: The run with Basilisk's.
    """
    assert list(compiled) == list(basilisk), f"the datasets differ: {list(compiled)} and {list(basilisk)}"
    for dataset, (values, units) in compiled.items():
        reference, reference_units = basilisk[dataset]
        assert units == reference_units, f"{dataset} is in {units}, Basilisk's run in {reference_units}"
        assert values.dtype == reference.dtype, f"{dataset} is {values.dtype}, Basilisk's run {reference.dtype}"
        assert values.shape == reference.shape, f"{dataset} is {values.shape}, Basilisk's run {reference.shape}"
        differing = np.argwhere(values.view(np.uint64) != reference.view(np.uint64))
        assert not differing.size, (
            f"{dataset} first differs at {tuple(differing[0])}: {values[tuple(differing[0])]!r},"
            f" Basilisk's run {reference[tuple(differing[0])]!r}"
        )


@pytest.mark.parametrize("case", [0, 1])
@pytest.mark.parametrize("fidelity", ["lo", "med"])
@pytest.mark.parametrize("vehicle", sorted(VEHICLES))
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_run_is_bit_identical_to_one_with_basilisks_integrator(
    name: str,
    vehicle: str,
    fidelity: str,
    case: int,
    commanding: type,
    torque: Callable[[float], Any],
    monkeypatch: pytest.MonkeyPatch,
):
    """Every dataset of a dispersed case is bit for bit the same as with Basilisk's own integrator.

    A vehicle with a wheel is commanded a torque every tick.

    Args:
        name: The scenario's name in `SCENARIOS`.
        vehicle: The vehicle's name in `VEHICLES`.
        fidelity: The fidelity, whose compiled integrator is compared.
        case: The dispersed case.
        commanding: The stub build class.
        torque: Builds a wheel torque command.
        monkeypatch: Swaps Basilisk's integrator in for the fidelity's.
    """

    def fly() -> dict:
        """Fly the case for 300 s.

        Returns:
            The run's datasets.
        """
        flown = SCENARIOS[name](VEHICLES[vehicle]())
        flown.fidelity = fidelity
        flown.disperse_case(np.random.SeedSequence(0, spawn_key=(case,)))
        commands = {"wheel": torque(0.02)} if "wheel" in flown.vehicle.writers() else {}
        return flown.run(FswHost(commanding(commands), flown.vehicle), {}, duration_s=300.0)

    compiled = fly()
    with monkeypatch.context() as patch:
        patch.setitem(scenario._FIDELITY, fidelity, _BASILISK[fidelity])
        basilisk = fly()

    _assert_bit_identical(compiled, basilisk)


class _WithMovingMoon(Leo500):
    """Leo500 with a Moon moving at 1 km/s, whose pull on the vehicle so depends on the time.

    The equations of motion of the other scenarios do not read the time, so only here do the stage
    times of a tableau, `c`, reach the data.
    """

    def __init__(self, vehicle: Any) -> None:
        """Place `vehicle` as Leo500 does, then add the Moon.

        Args:
            vehicle: The vehicle to fly.
        """
        super().__init__(vehicle)
        moon = self._gravity.createMoon()
        state = messaging.SpicePlanetStateMsgPayload()  # ty: ignore[not-callable]
        state.PositionVector = [3.84e8, 0.0, 0.0]
        state.VelocityVector = [0.0, 1.0e3, 0.0]
        self._moon_state = messaging.SpicePlanetStateMsg().write(state)
        moon.planetBodyInMsg.subscribeTo(self._moon_state)
        self._gravity.addBodiesTo(vehicle.spacecraft)


@pytest.mark.parametrize("fidelity", ["lo", "med"])
def test_a_run_whose_dynamics_depend_on_the_time_is_bit_identical_too(fidelity: str, monkeypatch: pytest.MonkeyPatch):
    """With a moving Moon, every dataset is bit for bit the same as with Basilisk's own integrator.

    Args:
        fidelity: The fidelity, whose compiled integrator is compared.
        monkeypatch: Swaps Basilisk's integrator in for the fidelity's.
    """

    def fly() -> dict:
        """Fly 300 s of the IMU spacecraft beside the moving Moon.

        Returns:
            The run's datasets.
        """
        flown = _WithMovingMoon(ImuSpacecraft())
        flown.fidelity = fidelity
        return flown.run(Guidance(), {}, duration_s=300.0)

    compiled = fly()
    with monkeypatch.context() as patch:
        patch.setitem(scenario._FIDELITY, fidelity, _BASILISK[fidelity])
        basilisk = fly()

    _assert_bit_identical(compiled, basilisk)


@pytest.mark.parametrize(("fidelity", "compiled"), [("lo", True), ("med", True), ("hi", False)])
def test_lo_and_med_integrate_with_the_compiled_integrator_and_hi_with_basilisks(fidelity: str, compiled: bool):
    """The spacecraft of a run at `lo` or `med` integrates with the compiled integrator; at `hi`, with Basilisk's.

    Args:
        fidelity: The fidelity the run flies at.
        compiled: Whether that fidelity's integrator is the compiled one.
    """
    flown = Leo500(ImuSpacecraft())
    flown.fidelity = fidelity

    flown.run(Guidance(), {}, duration_s=1.0)

    assert runge_kutta.integrates_here(flown.vehicle.spacecraft) is compiled, f"{fidelity} runs the wrong integrator"


@pytest.mark.parametrize("fidelity", ["lo", "med"])
@pytest.mark.parametrize("vehicle", sorted(VEHICLES))
def test_a_step_of_the_compiled_integrator_makes_no_native_allocation(
    vehicle: str, fidelity: str, tmp_path: Path, commanding: type
):
    """Under the integrator's step, only the Basilisk functions it calls allocate, by their own doing.

    Args:
        vehicle: The vehicle's name in `VEHICLES`.
        fidelity: The fidelity, whose compiled integrator is held.
        tmp_path: The test's own temporary directory.
        commanding: The stub build class.
    """

    def fly(ticks: int) -> int:
        """Fly `ticks` ticks of the vehicle.

        Args:
            ticks: The number of ticks to fly.

        Returns:
            How many times the spacecraft was stepped.
        """
        flown = SCENARIOS["leo500"](VEHICLES[vehicle]())
        flown.fidelity = fidelity
        build = commanding({})
        flown.run(FswHost(build, flown.vehicle), {}, duration_s=ticks * build.period_s)
        return flown.vehicle.spacecraft.CallCounts

    assert_no_native_allocations_per_tick(
        fly,
        within="wise_dynamics::RungeKutta::integrate",
        excluding=_BASILISK_ALLOCATING_IN_A_STEP,
        ticks=100,
        directory=tmp_path,
    )


def test_a_library_built_against_another_basilisk_is_refused(monkeypatch: pytest.MonkeyPatch):
    """Loading the library under a Basilisk of another source revision raises, naming both.

    Args:
        monkeypatch: Gives Basilisk another source revision.
    """
    monkeypatch.setattr(Basilisk, "getBuildInfo", lambda: {"artifact": {"sourceRevision": "0" * 40}})

    with pytest.raises(ImportError, match="built against Basilisk [0-9a-f]{40}, but Basilisk 0{40} is installed"):
        runge_kutta._load()


def test_the_library_reports_exactly_the_build_facts_held():
    """The library reports each held build fact, and no fact the refusal test below does not hold."""
    assert sorted(json.loads(runge_kutta._LIBRARY.wd_abi())) == _BUILD_FACTS


@pytest.mark.parametrize("fact", _BUILD_FACTS)
def test_a_library_built_with_other_build_facts_than_basilisks_is_refused(fact: str, monkeypatch: pytest.MonkeyPatch):
    """Loading the library under a Basilisk recording any build fact other than the library's raises, naming it.

    Args:
        fact: The dotted key, in `Basilisk.getBuildInfo()["abi"]`, of the fact Basilisk records differently.
        monkeypatch: Gives Basilisk the other fact.
    """
    info: Any = Basilisk.getBuildInfo()
    *path, leaf = fact.split(".")
    recorded = info["abi"]
    for key in path:
        recorded = recorded[key]
    recorded[leaf] = f"{recorded[leaf]}-other" if isinstance(recorded[leaf], str) else recorded[leaf] + 1
    monkeypatch.setattr(Basilisk, "getBuildInfo", lambda: info)

    with pytest.raises(ImportError, match=f"'{fact}'"):
        runge_kutta._load()


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [
        (((0.0, 0.0), (0.5,)), (0.5, 0.5), (0.0, 0.5)),
        (((0.0, 0.0),), (0.5, 0.5), (0.0, 0.5)),
        (((0.0, 0.0), (0.5, 0.0)), (0.5, 0.5), (0.0,)),
        (((0.0, 0.5), (0.5, 0.0)), (0.5, 0.5), (0.0, 0.5)),
        (((0.5, 0.0), (0.5, 0.0)), (0.5, 0.5), (0.0, 0.5)),
        ((), (), ()),
    ],
    ids=["short row", "too few rows", "short c", "above the diagonal", "on the diagonal", "no stage"],
)
def test_a_tableau_that_is_not_an_explicit_method_is_refused(a: tuple, b: tuple, c: tuple):
    """A tableau with a short row or array, a coefficient on or above the diagonal, or no stage raises.

    Args:
        a: The stage coefficients.
        b: The weights.
        c: The nodes.
    """
    with pytest.raises(ValueError, match="tableau|row"):
        runge_kutta.Tableau(a=a, b=b, c=c)


def test_a_spacecraft_that_does_not_take_the_integrator_is_refused():
    """A spacecraft whose integration is synced to another's keeps its integrator, and installing says so."""
    primary, secondary = bsk_spacecraft.Spacecraft(), bsk_spacecraft.Spacecraft()
    primary.syncDynamicsIntegration(secondary)

    with pytest.raises(RuntimeError, match="did not take the compiled integrator"):
        runge_kutta.install(secondary, scenario._RK4)

    assert not runge_kutta.integrates_here(secondary), "the synced spacecraft integrates with the compiled integrator"
