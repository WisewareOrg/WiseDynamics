"""Stubs the tests share."""

from Basilisk.fswAlgorithms import inertial3D


class Guidance(inertial3D.inertial3D):
    """Basilisk's own C++ `inertial3D` standing in for flight software at 1 Hz, so no call crosses into Python.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 1.0
