"""Runs a plant with Basilisk's stock `inertial3D` guidance module standing in for flight
software."""

from typing import Any

from Basilisk.fswAlgorithms import inertial3D

from wise_dynamics.basilisk import Plant, run_case

GUIDANCE_SIGMA_RN = [0.1, 0.2, 0.3]


def run_with_guidance(
    plant: Plant, duration_s: float, record_s: float = 10.0
) -> dict[str, tuple[Any, str]]:
    """Run `plant` with an `inertial3D` holding `GUIDANCE_SIGMA_RN`, recording the plant's
    fields and the guidance output under `guidance/sigma_RN`."""
    guidance = inertial3D.inertial3D()
    guidance.ModelTag = "guidance"
    guidance.sigma_R0N = GUIDANCE_SIGMA_RN
    fields = {
        **plant.recorded_fields(),
        "guidance/sigma_RN": (guidance.attRefOutMsg, "sigma_RN", "MRP"),
    }
    return run_case(plant, guidance, fields=fields, duration_s=duration_s, record_s=record_s)
