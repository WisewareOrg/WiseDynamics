"""Checks `run_case`: a stub flight-software module stands in for a real one, and a stock
Basilisk module proves `run_case` accepts any `SysModel`, given its output messages
explicitly."""

import numpy as np
from Basilisk.fswAlgorithms import inertial3D
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

    recorders = run_case(plant, stub, {"stub_out": stub.output_msg}, duration_s=50.0, record_s=10.0)

    assert set(recorders) == {"truth", "imu", "stub_out"}
    n = len(recorders["truth"].times())
    assert n > 0
    assert np.array(recorders["truth"].sigma_BN).shape == (n, 3)
    assert np.array(recorders["truth"].r_BN_N).shape == (n, 3)
    assert np.array(recorders["imu"].AngVelPlatform).shape == (n, 3)
    stub_sigma = np.array(recorders["stub_out"].sigma_BN)
    assert stub_sigma.shape == (n, 3)
    assert np.allclose(stub_sigma[-1], StubFsw.FIXED_SIGMA_BN)


def test_run_case_accepts_a_stock_basilisk_module_with_no_outputs_attribute():
    """`inertial3D.inertial3D()` is a real Basilisk FSW module, usable here precisely because
    `run_case` needs nothing from a flight-software module beyond its output messages, which
    the caller supplies explicitly."""
    sim = SimulationBaseClass.SimBaseClass()
    plant = ImuSpacecraftModel(sim)
    stock = inertial3D.inertial3D()
    stock.ModelTag = "inertial3D"
    stock.sigma_R0N = [0.0, 0.0, 0.0]

    recorders = run_case(
        plant, stock, {"att_ref": stock.attRefOutMsg}, duration_s=20.0, record_s=10.0
    )

    assert "att_ref" in recorders
    n = len(recorders["truth"].times())
    assert np.array(recorders["att_ref"].sigma_RN).shape == (n, 3)
