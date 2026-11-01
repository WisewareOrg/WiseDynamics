"""Checks `ImuSpacecraftModel`'s `rng_seed`: it seeds the IMU's own noise draw, so two cases with
different seeds record different sensor noise, and two cases with the same seed reproduce it
exactly."""

import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros

from sim.basilisk.models.imu_spacecraft import ImuSpacecraftModel


def _gyro_noise(rng_seed: int) -> np.ndarray:
    sim = SimulationBaseClass.SimBaseClass()
    model = ImuSpacecraftModel(
        sim,
        omega0_BN_B=np.zeros(3),
        gyro_bias=np.zeros(3),
        accel_bias=np.zeros(3),
        rng_seed=rng_seed,
    )
    imu_log = model.imuObject.sensorOutMsg.recorder(macros.sec2nano(1.0))
    sim.AddModelToTask(model.task_name, imu_log)

    sim.InitializeSimulation()
    sim.ConfigureStopTime(macros.sec2nano(50.0))
    sim.ExecuteSimulation()

    return np.array(imu_log.AngVelPlatform)


def test_different_seeds_record_different_noise():
    assert not np.allclose(_gyro_noise(0), _gyro_noise(1))


def test_same_seed_reproduces_identical_noise():
    assert np.array_equal(_gyro_noise(7), _gyro_noise(7))
