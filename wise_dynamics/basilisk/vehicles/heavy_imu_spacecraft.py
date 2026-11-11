"""The IMU spacecraft, heavier and differently shaped: a stand-in second vehicle."""

from wise_dynamics.basilisk.vehicles.imu_spacecraft import ImuSpacecraft

__all__ = ["HeavyImuSpacecraft"]


class HeavyImuSpacecraft(ImuSpacecraft):
    """`ImuSpacecraft` at three times the mass and x and y inertia, twice the z inertia, same IMU.

    Scaling every inertia by one factor would leave the torque-free tumble unchanged, and point-mass
    gravity ignores mass, so z scales less to make the vehicle fly differently.

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
