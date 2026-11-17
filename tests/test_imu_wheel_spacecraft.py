"""`ImuWheelSpacecraft`: the IMU spacecraft with one reaction wheel, commanded through its writer.

The tests cover its nominal wheel, how its keywords reach Basilisk's wheel, its writer and what it
records, and a commanded torque spinning the wheel as Basilisk's wheel model says, its spin momentum
recorded in truth.
"""

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from Basilisk.architecture import messaging

from wise_dynamics.basilisk import FswHost, ImuSpacecraft
from wise_dynamics.basilisk.scenarios.leo500 import Leo500
from wise_dynamics.basilisk.vehicles import VEHICLES
from wise_dynamics.basilisk.vehicles.imu_wheel_spacecraft import ImuWheelSpacecraft

TORQUE_N_M = 0.1
"""The commanded wheel torque, below the wheel's limit."""


def test_it_is_registered_and_is_the_imu_spacecraft_with_a_nominal_wheel():
    """Read from the class: the IMU spacecraft's values, and the wheel's nominal parameters."""
    assert VEHICLES["imu_wheel_spacecraft"] is ImuWheelSpacecraft
    assert issubclass(ImuWheelSpacecraft, ImuSpacecraft)
    assert ImuWheelSpacecraft.mass_kg == ImuSpacecraft.mass_kg
    assert ImuWheelSpacecraft.wheel_spin_axis_B == (0.0, 0.0, 1.0)
    assert ImuWheelSpacecraft.wheel_max_momentum_N_m_s == 50.0
    assert ImuWheelSpacecraft.wheel_speed_rad_s == 0.0


def test_each_wheel_keyword_defaults_to_its_class_attribute_and_reaches_the_wheel():
    """A vehicle built with no keywords has the nominal wheel; each keyword reaches Basilisk's.

    `ImuSpacecraft`'s keywords pass through to it.
    """
    parameters = inspect.signature(ImuWheelSpacecraft).parameters
    wheel_keywords = ["wheel_spin_axis_B", "wheel_max_momentum_N_m_s", "wheel_speed_rad_s"]
    assert list(parameters) == [*wheel_keywords, "imu_spacecraft"]
    nominal = ImuWheelSpacecraft()
    for name in wheel_keywords:
        assert parameters[name].default is None, name
        assert getattr(nominal, name) == getattr(ImuWheelSpacecraft, name), name
    assert ImuWheelSpacecraft(mass_kg=500.0).spacecraft.hub.mHub == 500.0

    vehicle = ImuWheelSpacecraft(
        wheel_spin_axis_B=(1.0, 0.0, 0.0), wheel_max_momentum_N_m_s=100.0, wheel_speed_rad_s=10.0
    )
    [wheel] = vehicle.wheel.ReactionWheelData
    omega_max = 6000.0 * 2.0 * np.pi / 60.0
    assert np.array_equal(np.array(wheel.gsHat_B).flatten(), (1.0, 0.0, 0.0))
    assert wheel.Js == pytest.approx(100.0 / omega_max, rel=1e-12)
    assert wheel.Omega == pytest.approx(10.0, rel=1e-12)
    assert wheel.RWModel == messaging.BalancedWheels


def test_the_wheel_writer_is_the_message_the_wheel_takes_its_command_from():
    """`writers()` hands out `wheel_command`, the same object each call, which the wheel reads."""
    vehicle = ImuWheelSpacecraft()
    assert vehicle.writers() == {"wheel": vehicle.wheel_command}
    assert vehicle.writers()["wheel"] is vehicle.writers()["wheel"]
    assert vehicle.wheel.rwMotorCmdInMsg.isLinked()
    assert vehicle.models() == [vehicle.wheel, vehicle.spacecraft]
    assert vehicle.sensor_models(0.5) == [vehicle.imu]
    assert set(vehicle.readers()) == {"imu"}


def test_it_records_the_imu_spacecraft_s_fields_and_the_wheel_speed():
    """`truth/wheel_speed`, rad/s, is the hub-written `RWSpeedMsg`, beside every inherited field."""
    vehicle = ImuWheelSpacecraft()
    fields = vehicle.recorded_fields()
    inherited = ImuSpacecraft.recorded_fields(vehicle)
    assert list(fields) == [*inherited, "truth/wheel_speed"]
    for path, (_, field, units) in inherited.items():
        assert fields[path][1:] == (field, units), path
    assert fields["truth/wheel_speed"][1:] == ("wheelSpeeds", "rad/s")


def _wheel_momentum(datasets: Mapping[str, tuple[np.ndarray, str]], vehicle: Any) -> np.ndarray:
    """The wheel's spin momentum about its axis at each sample: Js times its inertial spin rate.

    Args:
        datasets: A run's datasets.
        vehicle: The vehicle flown.

    Returns:
        Js * (Omega + gsHat_B . omega_BN_B), in N*m*s, per sample.
    """
    [wheel] = vehicle.wheel.ReactionWheelData
    omega, _ = datasets["truth/wheel_speed"]
    rate, _ = datasets["truth/omega_BN_B"]
    axis = np.array(wheel.gsHat_B).flatten()
    return wheel.Js * (omega + rate @ axis)


def test_a_commanded_torque_spins_the_wheel_as_basilisk_s_wheel_model_says(
    commanding: Callable[..., Any], torque: Callable[[float], Any]
):
    """The wheel's spin momentum gains the torque times the time since the tick that wrote it.

    For a balanced wheel Basilisk integrates Js (dOmega/dt + gsHat . domega/dt) = u, so the spin
    momentum Js (Omega + gsHat . omega_BN_B) gains u per second from the first tick, at 0.5 s:
    the command acts from the tick that wrote it, and the recorded speed is that of its instant.

    Args:
        commanding: The stub build that commands what it is given.
        torque: Builds a wheel torque command.
    """
    scenario = Leo500(vehicle_type=ImuWheelSpacecraft)
    host = FswHost(commanding({"wheel": torque(TORQUE_N_M)}), scenario.vehicle)

    datasets = scenario.run(host, {}, duration_s=4.0)

    t_s, _ = datasets["t_s"]
    omega, units = datasets["truth/wheel_speed"]
    assert units == "rad/s"
    assert omega.shape == t_s.shape
    assert np.all(omega[t_s > 0.5] > 0.0)
    momentum, units = datasets["truth/wheel_momentum"]
    assert units == "N*m*s"
    assert momentum.shape == t_s.shape
    np.testing.assert_allclose(momentum, _wheel_momentum(datasets, scenario.vehicle), rtol=1e-12)
    expected = TORQUE_N_M * np.clip(t_s - 0.5, 0.0, None)
    np.testing.assert_allclose(momentum - momentum[0], expected, rtol=1e-9, atol=1e-12)


def test_a_tick_with_no_command_leaves_the_last_one_in_force(
    commanding: Callable[..., Any], torque: Callable[[float], Any]
):
    """Commanded at the first tick only, the wheel keeps its torque for the rest of the run.

    Args:
        commanding: The stub build that commands what it is given.
        torque: Builds a wheel torque command.
    """
    scenario = Leo500(vehicle_type=ImuWheelSpacecraft)
    host = FswHost(commanding({"wheel": torque(TORQUE_N_M)}, ticks=1), scenario.vehicle)

    datasets = scenario.run(host, {}, duration_s=4.0)

    t_s, _ = datasets["t_s"]
    momentum, _ = datasets["truth/wheel_momentum"]
    np.testing.assert_allclose(
        momentum - momentum[0], TORQUE_N_M * np.clip(t_s - 0.5, 0.0, None), rtol=1e-9, atol=1e-12
    )
