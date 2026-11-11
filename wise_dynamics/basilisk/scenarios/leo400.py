"""An IMU spacecraft in a 400 km, 45 degree low Earth orbit under point-mass Earth gravity."""

from wise_dynamics.basilisk.scenarios.leo500 import Leo500

__all__ = ["Leo400"]


class Leo400(Leo500):
    """`Leo500` on a lower orbit: the same vehicle, environment, placement and dispersions.

    Attributes:
        semi_major_axis_m: Semi-major axis of the initial orbit, in metres.
    """

    semi_major_axis_m: float = 6771.0e3
