"""`Scenario.run`: what a run records, as run-file datasets, and that it runs once."""

import numpy as np
import pytest
from Basilisk.architecture import sysModel
from Basilisk.fswAlgorithms import inertial3D

from wise_dynamics.basilisk import Leo500

GUIDANCE_SIGMA_RN = [0.1, 0.2, 0.3]


class _Guidance(inertial3D.inertial3D):
    """Basilisk's stock `inertial3D` standing in for flight software, stepping at 1 Hz."""

    period_s = 1.0


def _guidance():
    """Basilisk's stock `inertial3D` standing in for flight software, and its field to record."""
    guidance = _Guidance()
    guidance.sigma_R0N = GUIDANCE_SIGMA_RN
    return guidance, {"guidance/sigma_RN": (guidance.attRefOutMsg, "sigma_RN", "MRP")}


def test_a_run_records_the_vehicle_and_fsw_fields_on_the_record_cadence():
    scenario = Leo500()
    fields = scenario.vehicle.recorded_fields()
    guidance, fsw_fields = _guidance()
    datasets = scenario.run(guidance, fsw_fields, duration_s=50.0)
    t_s, units = datasets["t_s"]
    assert units == "s"
    assert np.array_equal(t_s, np.arange(0.0, 50.0 + 10.0, 10.0))
    assert list(datasets) == ["t_s", *fields, "guidance/sigma_RN"]
    for name, (_, _, field_units) in fields.items():
        assert datasets[name][0].shape == (6, 3)
        assert datasets[name][1] == field_units
    sigma_RN, units = datasets["guidance/sigma_RN"]
    assert units == "MRP"
    assert np.allclose(sigma_RN[-1], GUIDANCE_SIGMA_RN)


def test_a_scenario_runs_once():
    scenario = Leo500()
    scenario.run(_Guidance(), {}, duration_s=1.0)
    with pytest.raises(RuntimeError, match="already run"):
        scenario.run(_Guidance(), {}, duration_s=1.0)


class _Stepping(sysModel.SysModel):
    """A flight-software module that notes each time it is called."""

    period_s = 1.0

    def __init__(self) -> None:
        super().__init__()
        self.ModelTag = "stepping"
        self.called_ns: list[int] = []

    def UpdateState(self, CurrentSimNanos: int) -> None:
        self.called_ns.append(CurrentSimNanos)


def test_the_flight_software_steps_at_its_own_period():
    fsw = _Stepping()
    Leo500().run(fsw, {}, duration_s=2.0)
    assert fsw.called_ns == [0, 1_000_000_000, 2_000_000_000]
