"""The IMU spacecraft carrying one reaction wheel, commanded through its writer by role `wheel`.

The wheel is Basilisk's `reactionWheelStateEffector` holding one Honeywell HR16 from Basilisk's
`rwFactory`, a balanced wheel at the hub's centre of mass.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from Basilisk.architecture import messaging
from Basilisk.simulation import reactionWheelStateEffector
from Basilisk.utilities import macros, simIncludeRW

from wise_dynamics.basilisk.vehicles.imu_spacecraft import ImuSpacecraft, Vector3

__all__ = ["ImuWheelSpacecraft"]


class ImuWheelSpacecraft(ImuSpacecraft):
    """`ImuSpacecraft` with one reaction wheel, which a flight software commands by role `wheel`.

    The wheel is a balanced Honeywell HR16 from Basilisk's `rwFactory`, at the hub's centre of mass:
    its spin inertia Js is its maximum momentum over its maximum speed, and its torque and speed
    limits are the factory's HR16 values. The wheel is not dispersed: `disperse` draws only the
    IMU's biases, as `ImuSpacecraft`'s does. Besides `ImuSpacecraft`'s, a vehicle holds `wheel`, the
    Basilisk `ReactionWheelStateEffector`, and `wheel_command`, the stand-alone
    `ArrayMotorTorqueMsg` the wheel reads its torque from, its first element. Its run file holds
    the wheel's speed, `truth/wheel_speed`, and its spin momentum, `truth/wheel_momentum`.

    Attributes:
        wheel_spin_axis_B: Nominal spin axis of the wheel in body axes, a unit vector.
        wheel_max_momentum_N_m_s: Nominal maximum momentum of the wheel, in N*m*s: 50, 75 or 100,
            the HR16's sizes.
        wheel_speed_rad_s: Nominal initial speed of the wheel relative to the hub, in rad/s.
    """

    wheel_spin_axis_B: Vector3 = (0.0, 0.0, 1.0)
    wheel_max_momentum_N_m_s: float = 50.0
    wheel_speed_rad_s: float = 0.0

    def __init__(
        self,
        *,
        wheel_spin_axis_B: Vector3 | None = None,
        wheel_max_momentum_N_m_s: float | None = None,
        wheel_speed_rad_s: float | None = None,
        **imu_spacecraft: Any,
    ) -> None:
        """Build the hub and the IMU as `ImuSpacecraft` does, then the wheel and its command.

        Args:
            wheel_spin_axis_B: Spin axis of the wheel in body axes, a unit vector.
            wheel_max_momentum_N_m_s: Maximum momentum of the wheel, in N*m*s: 50, 75 or 100.
            wheel_speed_rad_s: Initial speed of the wheel relative to the hub, in rad/s.
            **imu_spacecraft: `ImuSpacecraft`'s keywords.
        """
        super().__init__(**imu_spacecraft)
        nominal = type(self)
        self.wheel_spin_axis_B = nominal.wheel_spin_axis_B if wheel_spin_axis_B is None else wheel_spin_axis_B
        self.wheel_max_momentum_N_m_s = (
            nominal.wheel_max_momentum_N_m_s if wheel_max_momentum_N_m_s is None else wheel_max_momentum_N_m_s
        )
        self.wheel_speed_rad_s = nominal.wheel_speed_rad_s if wheel_speed_rad_s is None else wheel_speed_rad_s

        self._wheels = simIncludeRW.rwFactory()
        self._wheels.create(
            "Honeywell_HR16",
            list(self.wheel_spin_axis_B),
            maxMomentum=float(self.wheel_max_momentum_N_m_s),
            Omega=float(self.wheel_speed_rad_s / macros.RPM),
            RWModel=messaging.BalancedWheels,
        )
        self.wheel = reactionWheelStateEffector.ReactionWheelStateEffector()
        self._wheels.addToSpacecraft("wheel", self.wheel, self.spacecraft)
        self.wheel_command = messaging.ArrayMotorTorqueMsg()
        self.wheel.rwMotorCmdInMsg.subscribeTo(self.wheel_command)

    def models(self) -> list[Any]:
        """List the vehicle's propagated Basilisk modules in task order.

        The wheel runs first, as in Basilisk's own examples, so it reads the latest command before
        the hub integrates the step: a command written at a tick acts from that tick.

        Returns:
            The wheel, then the hub.
        """
        return [self.wheel, self.spacecraft]

    def writers(self) -> dict[str, Any]:
        """Hand out the vehicle's actuator inputs by role.

        Returns:
            Role `wheel`: `wheel_command`, whose payload is an `ArrayMotorTorqueMsgPayload` whose
            first `motorTorque` is the wheel's torque, in N*m.
        """
        return {"wheel": self.wheel_command}

    def recorded_fields(self) -> dict[str, tuple[Any, str, str]]:
        """Declare what the vehicle records.

        `truth/wheel_speed` is recorded from the `RWSpeedMsg` the hub writes once it has
        integrated, so it is the speed of the recorded instant; the wheel's own `RWConfigLogMsg`
        is written before the hub integrates, a step behind.

        Returns:
            `ImuSpacecraft`'s fields, and `truth/wheel_speed`: the wheel's speed relative to the
            hub, in rad/s, the `RWSpeedMsg`'s `wheelSpeeds`, which `finish` cuts to the one wheel.
        """
        return {
            **super().recorded_fields(),
            "truth/wheel_speed": (self.wheel.rwSpeedOutMsg, "wheelSpeeds", "rad/s"),
        }

    def finish(self, datasets: Mapping[str, tuple[Any, str]]) -> dict[str, tuple[Any, str]]:
        """Finish as `ImuSpacecraft` does, keep the one wheel's speed, and add its spin momentum.

        Args:
            datasets: Dataset name to `(array, units)`, as the recorders captured them.

        Returns:
            `ImuSpacecraft`'s finished datasets, with `truth/wheel_speed` cut from the
            `RWSpeedMsg`'s (N, 36) array to the wheel's (N,) speed, and `truth/wheel_momentum`, the
            wheel's angular momentum about its spin axis, Js (Omega + gsHat_B . omega_BN_B), (N,)
            in N*m*s.
        """
        finished = super().finish(datasets)
        speeds, units = finished["truth/wheel_speed"]
        rate, _ = finished["truth/omega_BN_B"]
        [wheel] = self.wheel.ReactionWheelData
        axis = np.array(wheel.gsHat_B).flatten()
        finished["truth/wheel_speed"] = (speeds[:, 0], units)
        finished["truth/wheel_momentum"] = (wheel.Js * (speeds[:, 0] + rate @ axis), "N*m*s")
        return finished
