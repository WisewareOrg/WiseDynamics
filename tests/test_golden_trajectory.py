"""Regression test: pins `ImuSpacecraftModel`'s recorded truth trajectory at t=300s against a
stored reference (`tests/golden/imu_spacecraft_300s.json`), so a Basilisk upgrade that silently
changes the plant's physics is caught rather than passed through unnoticed.

This test is expected to fail on a Basilisk upgrade that changes integration or gravity
behavior enough to move the trajectory outside tolerance. When that happens, updating the
golden reference is a deliberate act: regenerate the file and record the change -- the new
`bsk` version and why the trajectory moved -- in that update's own PR. Never update the
reference just to make this test pass without first understanding why it moved.
"""

import json
from pathlib import Path

import numpy as np
from Basilisk.utilities import SimulationBaseClass

from sim.basilisk.models.imu_spacecraft import ImuSpacecraftModel
from sim.basilisk.run import run_case
from tests.stubs import StubFsw

GOLDEN_PATH = Path(__file__).parent / "golden" / "imu_spacecraft_300s.json"

POSITION_TOL_M = 1e-3
VELOCITY_TOL_M_S = 1e-6
ATTITUDE_TOL_MRP = 1e-9


def test_truth_trajectory_matches_golden_reference():
    sim = SimulationBaseClass.SimBaseClass()
    plant = ImuSpacecraftModel(sim, gyro_noise_std=0.0, accel_noise_std=0.0)
    stub = StubFsw()
    stub.ModelTag = "stub"

    recorders = run_case(plant, stub, duration_s=300.0, record_s=10.0)

    golden = json.loads(GOLDEN_PATH.read_text())

    final_r_N = np.array(recorders["truth"].r_BN_N)[-1]
    final_v_N = np.array(recorders["truth"].v_BN_N)[-1]
    final_sigma_BN = np.array(recorders["truth"].sigma_BN)[-1]

    # rtol=0.0: these are absolute tolerances. np.allclose's default rtol=1e-5 would otherwise
    # dominate on r_N's ~1e6 m magnitude (1e-5 * 1e6 m = 10 m, far looser than the stated 1e-3 m)
    # and silently pass a moved reference -- confirmed by seeding a 1 m perturbation and seeing
    # it pass before this fix.
    assert np.allclose(final_r_N, golden["r_N"], atol=POSITION_TOL_M, rtol=0.0), (
        f"r_N {final_r_N} vs golden {golden['r_N']}"
    )
    assert np.allclose(final_v_N, golden["v_N"], atol=VELOCITY_TOL_M_S, rtol=0.0), (
        f"v_N {final_v_N} vs golden {golden['v_N']}"
    )
    assert np.allclose(final_sigma_BN, golden["sigma_BN"], atol=ATTITUDE_TOL_MRP, rtol=0.0), (
        f"sigma_BN {final_sigma_BN} vs golden {golden['sigma_BN']}"
    )
