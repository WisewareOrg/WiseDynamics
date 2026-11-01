"""Pins `ImuSpacecraftModel`'s stable interface: the keys `initial_state()` returns, and the
roles `readers()` and `truth_messages()` report, as `sim/basilisk/README.md` states them. A
consumer's own host adapter codes against these names directly."""

from Basilisk.utilities import SimulationBaseClass

from sim.basilisk.models.imu_spacecraft import ImuSpacecraftModel


def test_initial_state_returns_exactly_these_keys():
    sim = SimulationBaseClass.SimBaseClass()
    model = ImuSpacecraftModel(sim)
    assert set(model.initial_state()) == {"sigma0_BN", "r0_N", "v0_N", "omega0_BN_B"}


def test_readers_has_role_imu():
    sim = SimulationBaseClass.SimBaseClass()
    model = ImuSpacecraftModel(sim)
    assert set(model.readers()) == {"imu"}


def test_truth_messages_has_roles_truth_and_imu():
    sim = SimulationBaseClass.SimBaseClass()
    model = ImuSpacecraftModel(sim)
    assert set(model.truth_messages()) == {"truth", "imu"}
