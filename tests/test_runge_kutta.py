"""The compiled Runge-Kutta: bit-identical to Basilisk's, allocating nothing per step, refusing what it cannot run."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import Basilisk
import numpy as np
import pytest
from Basilisk.architecture import messaging
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.simulation import spacecraft as bsk_spacecraft
from Basilisk.simulation import svIntegrators

from tests.allocations import native_allocations, native_allocations_by_function
from wise_dynamics.basilisk import FswHost, ImuSpacecraft, scenario
from wise_dynamics.basilisk import _runge_kutta as runge_kutta
from wise_dynamics.basilisk.scenarios import SCENARIOS
from wise_dynamics.basilisk.scenarios.leo500 import Leo500
from wise_dynamics.basilisk.vehicles import VEHICLES


def _basilisk_rk3(craft: Any) -> Any:
    """Give a spacecraft Basilisk's own three-stage Runge-Kutta with Ralston's tableau, written out here.

    The tableau is the literal `lo` integrated with before the compiled integrator, not
    `scenario.RALSTON_RK3`, so a change to that constant fails the comparison.

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


def _fly(name: str, vehicle: str, fidelity: str, case: int, commanding: type, torque: Callable[[float], Any]) -> dict:
    """Fly one dispersed case for 300 s, commanding a wheel, if the vehicle has one, every tick.

    Args:
        name: The scenario's name in `SCENARIOS`.
        vehicle: The vehicle's name in `VEHICLES`.
        fidelity: The fidelity.
        case: The case number.
        commanding: The stub build class.
        torque: Builds a wheel torque command.

    Returns:
        The run's datasets.
    """
    flown = SCENARIOS[name](VEHICLES[vehicle]())
    flown.fidelity = fidelity
    flown.disperse_case(np.random.SeedSequence(0, spawn_key=(case,)))
    commands = {"wheel": torque(0.02)} if "wheel" in flown.vehicle.writers() else {}
    return flown.run(FswHost(commanding(commands), flown.vehicle), {}, duration_s=300.0)


@pytest.mark.parametrize("fidelity", ["lo", "med"])
@pytest.mark.parametrize("vehicle", sorted(VEHICLES))
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_a_run_is_bit_identical_to_one_with_basilisks_integrator(
    name: str,
    vehicle: str,
    fidelity: str,
    commanding: type,
    torque: Callable[[float], Any],
    monkeypatch: pytest.MonkeyPatch,
):
    """Every dataset of two dispersed cases is bit for bit the same as with Basilisk's own integrator.

    Args:
        name: The scenario's name in `SCENARIOS`.
        vehicle: The vehicle's name in `VEHICLES`.
        fidelity: The fidelity, whose compiled integrator is compared.
        commanding: The stub build class.
        torque: Builds a wheel torque command.
        monkeypatch: Swaps Basilisk's integrator in for the fidelity's.
    """
    for case in range(2):
        compiled = _fly(name, vehicle, fidelity, case, commanding, torque)
        with monkeypatch.context() as patch:
            patch.setitem(scenario._FIDELITY, fidelity, _BASILISK[fidelity])
            basilisk = _fly(name, vehicle, fidelity, case, commanding, torque)
        assert list(compiled) == list(basilisk)
        for dataset, (values, units) in compiled.items():
            reference, reference_units = basilisk[dataset]
            assert units == reference_units
            assert values.dtype == reference.dtype and values.shape == reference.shape
            assert values.tobytes() == reference.tobytes(), dataset


class _Guidance(inertial3D.inertial3D):
    """Basilisk's stock `inertial3D` standing in for flight software, stepping at 1 Hz.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 1.0


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
        return flown.run(_Guidance(), {}, duration_s=300.0)

    compiled = fly()
    with monkeypatch.context() as patch:
        patch.setitem(scenario._FIDELITY, fidelity, _BASILISK[fidelity])
        basilisk = fly()
    assert list(compiled) == list(basilisk)
    for dataset, (values, units) in compiled.items():
        reference, reference_units = basilisk[dataset]
        assert units == reference_units
        assert values.dtype == reference.dtype and values.shape == reference.shape
        assert values.tobytes() == reference.tobytes(), dataset


@pytest.mark.parametrize("fidelity", ["lo", "med"])
def test_a_step_of_the_compiled_integrator_makes_no_native_allocation(fidelity: str, tmp_path: Path, commanding: type):
    """No allocation per step is the integrator's own; those of the equations of motion it calls are Basilisk's.

    The integrator does allocate once, as it binds its states at its first step, which shows memray
    names its frames.

    Args:
        fidelity: The fidelity, whose compiled integrator is held.
        tmp_path: The test's own temporary directory.
        commanding: The stub build class.
    """

    def fly(ticks: int) -> int:
        """Fly `ticks` ticks of the IMU spacecraft.

        Args:
            ticks: The number of ticks to fly.

        Returns:
            How many times the spacecraft was stepped.
        """
        flown = SCENARIOS["leo500"](ImuSpacecraft())
        flown.fidelity = fidelity
        build = commanding({})
        flown.run(FswHost(build, flown.vehicle), {}, duration_s=ticks * build.period_s)
        return flown.vehicle.spacecraft.CallCounts

    integrate = "wise_dynamics::RungeKutta::integrate"
    assert native_allocations(fly, within=integrate, ticks=10, directory=tmp_path) > 0
    per_tick = native_allocations_by_function(fly, ticks=100, directory=tmp_path)
    assert {function: n for function, n in per_tick.items() if function.startswith("wise_dynamics::")} == {}


def test_a_library_built_against_another_basilisk_is_refused(monkeypatch: pytest.MonkeyPatch):
    """Loading the library under a Basilisk of another source revision raises, naming both.

    Args:
        monkeypatch: Gives Basilisk another source revision.
    """
    monkeypatch.setattr(Basilisk, "getBuildInfo", lambda: {"artifact": {"sourceRevision": "0" * 40}})
    with pytest.raises(ImportError, match="built against Basilisk [0-9a-f]{40}, but Basilisk 0{40} is installed"):
        runge_kutta._load()


def test_a_library_built_with_another_layout_than_basilisks_is_refused(monkeypatch: pytest.MonkeyPatch):
    """Loading the library under a Basilisk recording another Eigen alignment raises, naming the fact.

    Args:
        monkeypatch: Gives Basilisk another recorded alignment.
    """
    info: Any = Basilisk.getBuildInfo()
    info["abi"]["dependencies"]["eigen"]["maxAlignBytes"] = 32
    monkeypatch.setattr(Basilisk, "getBuildInfo", lambda: info)
    with pytest.raises(ImportError, match=r"'dependencies.eigen.maxAlignBytes': \(16, 32\)"):
        runge_kutta._load()


@pytest.mark.parametrize(
    ("a", "b", "c"),
    [
        (((0.0, 0.0), (0.5,)), (0.5, 0.5), (0.0, 0.5)),
        (((0.0, 0.0), (0.5, 0.0)), (0.5, 0.5), (0.0,)),
        (((0.0, 0.5), (0.5, 0.0)), (0.5, 0.5), (0.0, 0.5)),
        ((), (), ()),
    ],
    ids=["short row", "short c", "implicit", "no stage"],
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
        runge_kutta.install(secondary, scenario.RK4)
