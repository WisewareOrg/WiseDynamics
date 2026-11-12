"""Each fidelity's truth stays within its budget of `hi` over one orbit of every scenario.

The budget bounds the truth position's distance from `hi`'s at every recorded instant: under 1 m at
`lo` and under 1 mm at `med`. A stand-in flight software at 1 Hz sets the step to 1 s, the largest
any tier takes.
"""

import functools

import numpy as np
import pytest
from Basilisk.fswAlgorithms import inertial3D

from wise_dynamics.basilisk.scenarios import SCENARIOS

ORBIT_S = 5700.0
BUDGET_M = {"lo": 1.0, "med": 1e-3}


class _Guidance(inertial3D.inertial3D):
    """Basilisk's stock `inertial3D` standing in for flight software, stepping at 1 Hz.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 1.0


@functools.cache
def _truth_position(scenario: str, fidelity: str) -> np.ndarray:
    """The nominal scenario's recorded truth position over one orbit at a fidelity.

    Args:
        scenario: The scenario's name in `SCENARIOS`.
        fidelity: The fidelity it runs at.

    Returns:
        The `truth/r_N` dataset, in metres.
    """
    built = SCENARIOS[scenario]()
    built.fidelity = fidelity
    return built.run(_Guidance(), {}, duration_s=ORBIT_S)["truth/r_N"][0]


@pytest.mark.parametrize("fidelity", sorted(BUDGET_M))
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_each_fidelity_stays_within_its_budget_of_hi(scenario: str, fidelity: str):
    """The truth position at every recorded instant is within the fidelity's budget of `hi`'s.

    Args:
        scenario: The scenario's name in `SCENARIOS`.
        fidelity: The fidelity under test.
    """
    difference = _truth_position(scenario, fidelity) - _truth_position(scenario, "hi")
    assert np.linalg.norm(difference, axis=1).max() < BUDGET_M[fidelity]
