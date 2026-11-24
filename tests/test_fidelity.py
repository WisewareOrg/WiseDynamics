"""Each fidelity's truth stays within its budget of `hi` over one orbit of every scenario.

`BUDGET_M` bounds the truth position's distance from `hi`'s at every recorded instant. A stand-in
flight software at 1 Hz sets the step to 1 s, the largest any fidelity takes.
"""

import functools

import numpy as np
import pytest

from tests.stubs import Guidance
from wise_dynamics.basilisk import ImuSpacecraft
from wise_dynamics.basilisk.scenarios import SCENARIOS

ORBIT_S = 5700.0
BUDGET_M = {"lo": 1.0, "med": 1e-3}


@functools.cache
def _truth_position(scenario: str, fidelity: str) -> np.ndarray:
    """The nominal scenario's recorded truth position over one orbit at a fidelity.

    Args:
        scenario: The scenario's name in `SCENARIOS`.
        fidelity: The fidelity it runs at.

    Returns:
        The `truth/r_N` dataset, in metres.
    """
    built = SCENARIOS[scenario](ImuSpacecraft())
    built.fidelity = fidelity
    return built.run(Guidance(), {}, duration_s=ORBIT_S)["truth/r_N"][0]


@pytest.mark.parametrize("fidelity", sorted(BUDGET_M))
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_each_fidelity_stays_within_its_budget_of_hi(scenario: str, fidelity: str):
    """The truth position at every recorded instant is within the fidelity's budget of `hi`'s.

    It also differs from `hi`'s somewhere, so a fidelity that integrates as `hi` does is caught.

    Args:
        scenario: The scenario's name in `SCENARIOS`.
        fidelity: The fidelity under test.
    """
    difference = _truth_position(scenario, fidelity) - _truth_position(scenario, "hi")
    distance = np.linalg.norm(difference, axis=1)
    assert 0.0 < distance.max() < BUDGET_M[fidelity]
