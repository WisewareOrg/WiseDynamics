"""An IMU spacecraft whose central body differs from Earth's in its gravitational parameter."""

from tests.support import EARTH_MU_M3_S2
from wise_dynamics.basilisk import ImuSpacecraft


class OtherMuSpacecraft(ImuSpacecraft):
    """`ImuSpacecraft` with `mu` 0.1% above Basilisk's Earth value, so a test can tell the class
    attribute apart from Basilisk's own default."""

    mu = EARTH_MU_M3_S2 * 1.001
