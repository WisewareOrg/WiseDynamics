"""The nominal plant's attitude and noiseless IMU output at t = 300 s against a stored reference.

`tests/golden/imu_spacecraft_300s.json` holds the reference. A Basilisk release that changes the
attitude dynamics or the IMU model fails this test. Regenerating the reference is a deliberate
change, recorded with its reason and the new `bsk` version in the pull request that makes it:

    uv run python -m tests.test_golden_trajectory
"""

import dataclasses
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from tests.support import run_with_stub
from wise_dynamics.basilisk import ImuSpacecraft

GOLDEN_PATH = Path(__file__).parent / "golden" / "imu_spacecraft_300s.json"
DURATION_S = 300.0

ATTITUDE_TOL_MRP = 1e-9
GYRO_TOL_RAD_S = 1e-12
ACCEL_TOL_M_S2 = 1e-12


def golden_state() -> dict[str, list[float]]:
    """The nominal, noiseless plant's state at `DURATION_S`: the quantities the reference holds."""
    config = dataclasses.replace(
        ImuSpacecraft.nominal(), gyro_noise_std_rad_s=0.0, accel_noise_std_m_s2=0.0
    )
    recording = run_with_stub(ImuSpacecraft(config), duration_s=DURATION_S)
    return {
        "sigma_BN": np.array(recording["truth"].sigma_BN)[-1].tolist(),
        "gyro_P": np.array(recording["imu"].AngVelPlatform)[-1].tolist(),
        "accel_P": np.array(recording["imu"].AccelPlatform)[-1].tolist(),
    }


def test_attitude_and_imu_output_match_the_golden_reference():
    golden = json.loads(GOLDEN_PATH.read_text())
    state = golden_state()
    versions = f"reference recorded with bsk {golden['_bsk_version']}, installed {version('bsk')}"

    # rtol=0.0 makes each tolerance absolute.
    for key, tolerance in (
        ("sigma_BN", ATTITUDE_TOL_MRP),
        ("gyro_P", GYRO_TOL_RAD_S),
        ("accel_P", ACCEL_TOL_M_S2),
    ):
        assert np.allclose(state[key], golden[key], rtol=0.0, atol=tolerance), (
            f"{key} {state[key]} vs golden {golden[key]}; {versions}"
        )


if __name__ == "__main__":
    reference = {
        "_bsk_version": version("bsk"),
        "_description": (
            "ImuSpacecraft nominal configuration with zero sensor noise, at t = 300 s. "
            "Written by `uv run python -m tests.test_golden_trajectory`."
        ),
        **golden_state(),
    }
    GOLDEN_PATH.write_text(json.dumps(reference, indent=2) + "\n")
    print(f"wrote {GOLDEN_PATH} with bsk {reference['_bsk_version']}")
