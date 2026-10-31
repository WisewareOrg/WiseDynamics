"""Model-level checks for `ImuSpacecraftModel`: the IMU's own configured noise shows up in its
output at the configured magnitude."""

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

from sim.basilisk.models.imu_spacecraft import (
    ACCEL_NOISE_STD_DEFAULT,
    GYRO_NOISE_STD_DEFAULT,
    ImuSpacecraftModel,
)


def test_gyro_and_accel_noise_std_matches_the_configured_sigma():
    """Zero bias, zero body rate: the recorded signal is noise alone, so its standard
    deviation should land within 30% of the configured 1-sigma. This is the check that would
    have caught the noise being configured eight orders of magnitude too small."""
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
        rel_err = abs(axis_std - GYRO_NOISE_STD_DEFAULT) / GYRO_NOISE_STD_DEFAULT
        assert rel_err < 0.30, f"gyro noise std {axis_std} vs configured {GYRO_NOISE_STD_DEFAULT}"
    for axis_std in accel_std:
        rel_err = abs(axis_std - ACCEL_NOISE_STD_DEFAULT) / ACCEL_NOISE_STD_DEFAULT
        assert rel_err < 0.30, f"accel noise std {axis_std} vs configured {ACCEL_NOISE_STD_DEFAULT}"
