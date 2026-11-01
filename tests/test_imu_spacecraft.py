"""Model-level checks for `ImuSpacecraftModel`: the IMU's own configured noise shows up in its
output at the magnitude this repository has settled on."""

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

from sim.basilisk.models.imu_spacecraft import ImuSpacecraftModel

# Pinned as literals rather than imported from imu_spacecraft.py, so this test fails if that
# module's own default ever drifts from this value.
GYRO_STD_SIGMA = 5e-5  # rad/s, 1-sigma
ACCEL_STD_SIGMA = 5e-5  # m/s^2, 1-sigma


def test_gyro_and_accel_noise_std_matches_the_settled_sigma():
    """Zero bias, zero body rate: the recorded signal is noise alone, so its standard
    deviation should land within 30% of the settled 1-sigma above."""
    sim = SimulationBaseClass.SimBaseClass()
    model = ImuSpacecraftModel(
        sim,
        omega0_BN_B=np.zeros(3),
        gyro_bias=np.zeros(3),
        accel_bias=np.zeros(3),
    )
    imu_log = model.imuObject.sensorOutMsg.recorder(macros.sec2nano(1.0))
    sim.AddModelToTask(model.task_name, imu_log)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(500.0))
    sim.ExecuteSimulation()

    gyro_std = np.array(imu_log.AngVelPlatform)[5:].std(axis=0)
    accel_std = np.array(imu_log.AccelPlatform)[5:].std(axis=0)

    for axis_std in gyro_std:
        rel_err = abs(axis_std - GYRO_STD_SIGMA) / GYRO_STD_SIGMA
        assert rel_err < 0.30, f"gyro noise std {axis_std} vs expected {GYRO_STD_SIGMA}"
    for axis_std in accel_std:
        rel_err = abs(axis_std - ACCEL_STD_SIGMA) / ACCEL_STD_SIGMA
        assert rel_err < 0.30, f"accel noise std {axis_std} vs expected {ACCEL_STD_SIGMA}"
