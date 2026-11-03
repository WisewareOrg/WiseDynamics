"""A rigid spacecraft in low Earth orbit carrying one body-mounted IMU.

The vehicle is a rigid hub under point-mass Earth gravity, started on a classical-element orbit
with an initial attitude and body rate, and carries an IMU at its centre of mass with identity
mounting, constant biases and white noise. `wise_dynamics/basilisk/README.md` tabulates its
parameters with their units and frames.
"""

from wise_dynamics.basilisk.plants.imu_spacecraft.config import ImuSpacecraftConfig
from wise_dynamics.basilisk.plants.imu_spacecraft.plant import ImuSpacecraft

__all__ = ["ImuSpacecraft", "ImuSpacecraftConfig"]
