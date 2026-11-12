"""The IMU spacecraft, heavier and differently shaped: a stand-in second vehicle."""

from wise_dynamics.basilisk.vehicles.imu_spacecraft import ImuSpacecraft

__all__ = ["HeavyImuSpacecraft"]


class HeavyImuSpacecraft(ImuSpacecraft):
    """`ImuSpacecraft` with a different mass and inertia ratio, so it tumbles differently; same IMU.

    Attributes:
        mass_kg: Nominal hub mass, in kilograms.
        inertia_B_kg_m2: Nominal hub inertia about its centre of mass in body axes, in kg*m^2.
    """

    mass_kg: float = 2250.0
    inertia_B_kg_m2 = (
        (2700.0, 0.0, 0.0),
        (0.0, 2400.0, 0.0),
        (0.0, 0.0, 1200.0),
    )
