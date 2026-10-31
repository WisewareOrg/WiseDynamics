"""Checks `run_case` without any algorithm: a stub flight-software module stands in for a real
one, so this repository can test its own runner without depending on any flight software."""

import numpy as np
from Basilisk.utilities import SimulationBaseClass

from sim.basilisk.models.imu_spacecraft import ImuSpacecraftModel
from sim.basilisk.run import run_case
from tests.stubs import StubFsw


def test_plant_builds():
    sim = SimulationBaseClass.SimBaseClass()
    plant = ImuSpacecraftModel(sim)
    assert plant.scObject is not None
    assert plant.imuObject is not None
    assert plant.mu > 0.0


def test_run_case_produces_arrays_of_the_right_shape():
    sim = SimulationBaseClass.SimBaseClass()
    plant = ImuSpacecraftModel(sim)
    stub = StubFsw()
    stub.ModelTag = "stub"

    recorders = run_case(plant, stub, duration_s=50.0, record_s=10.0)

    assert set(recorders) == {"truth", "imu", "stub_out"}
    n = len(recorders["truth"].times())
    assert n > 0
    assert np.array(recorders["truth"].sigma_BN).shape == (n, 3)
    assert np.array(recorders["truth"].r_BN_N).shape == (n, 3)
    assert np.array(recorders["imu"].AngVelPlatform).shape == (n, 3)
    stub_sigma = np.array(recorders["stub_out"].sigma_BN)
    assert stub_sigma.shape == (n, 3)
    assert np.allclose(stub_sigma[-1], StubFsw.FIXED_SIGMA_BN)
