"""`Scenario.run`: what a run records, as run-file datasets, and what it refuses."""

import numpy as np
import pytest
from Basilisk.fswAlgorithms import inertial3D

from wise_dynamics.basilisk import ImuSpacecraft, Leo500

GUIDANCE_SIGMA_RN = [0.1, 0.2, 0.3]


def _guidance():
    """Basilisk's stock `inertial3D` standing in for flight software, and its field to record."""
    guidance = inertial3D.inertial3D()
    guidance.sigma_R0N = GUIDANCE_SIGMA_RN
    return guidance, {"guidance/sigma_RN": (guidance.attRefOutMsg, "sigma_RN", "MRP")}


def test_a_run_records_the_vehicle_and_fsw_fields_on_the_record_cadence():
    vehicle = ImuSpacecraft()
    scenario = Leo500()
    scenario.add(vehicle)
    fields = vehicle.recorded_fields()
    guidance, fsw_fields = _guidance()
    datasets = scenario.run(guidance, fsw_fields=fsw_fields, duration_s=50.0, record_s=10.0)
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


@pytest.mark.parametrize(
    ("refusal", "error", "message"),
    [
        ("no vehicle", ValueError, "add a vehicle before running"),
        ("record_s", ValueError, "recording period must be positive"),
        ("duration_s", ValueError, "duration must be positive"),
        ("t_s", ValueError, r"reuses the run's own dataset paths \['t_s'\]"),
        ("truth/r_N", ValueError, r"reuses the run's own dataset paths \['truth/r_N'\]"),
        ("absent field", ValueError, "'bad': its message has no field 'r_BN_N'"),
        ("already run", RuntimeError, "already run"),
    ],
)
def test_a_refused_run_leaves_the_scenario_unchanged(refusal, error, message):
    """A refused run builds nothing and leaves a scenario that has not run able to run."""
    vehicle = ImuSpacecraft()
    scenario = Leo500()
    if refusal != "no vehicle":
        scenario.add(vehicle)
    guidance, fsw_fields = _guidance()
    arguments = {"fsw_fields": fsw_fields, "duration_s": 10.0, "record_s": 10.0}
    if refusal in ("record_s", "duration_s"):
        arguments[refusal] = 0.0
    elif refusal in ("t_s", "truth/r_N"):
        arguments["fsw_fields"] = {refusal: fsw_fields["guidance/sigma_RN"]}
    elif refusal == "absent field":
        arguments["fsw_fields"] = {"bad": (guidance.attRefOutMsg, "r_BN_N", "m")}
    elif refusal == "already run":
        scenario.run(inertial3D.inertial3D(), fsw_fields={}, duration_s=1.0)
    tasks_before = [list(task.TaskModels) for task in scenario._simulation.TaskList]

    with pytest.raises(error, match=message):
        scenario.run(guidance, **arguments)
    assert [list(task.TaskModels) for task in scenario._simulation.TaskList] == tasks_before
    assert len(vehicle.spacecraft.gravField.gravBodies) == (refusal == "already run")
    if refusal == "no vehicle":
        scenario.add(vehicle)
    if refusal != "already run":
        scenario.run(guidance, fsw_fields=fsw_fields, duration_s=10.0)
