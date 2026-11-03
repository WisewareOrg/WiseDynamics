"""`Scenario.run`: what a run records, as run-file datasets, and what it refuses."""

import numpy as np
import pytest
from Basilisk.fswAlgorithms import inertial3D

from wise_dynamics.basilisk import ImuSpacecraft, Leo500


def test_a_run_records_the_vehicle_fields_on_the_record_cadence():
    scenario = Leo500(ImuSpacecraft())
    fields = scenario.vehicle.recorded_fields()
    datasets = scenario.run(inertial3D.inertial3D(), duration_s=50.0, record_s=10.0)
    t_s, units = datasets["t_s"]
    assert units == "s"
    assert np.array_equal(t_s, np.arange(0.0, 50.0 + 10.0, 10.0))
    assert list(datasets) == ["t_s", *fields]
    for name, (_, _, field_units) in fields.items():
        assert datasets[name][0].shape == (6, 3)
        assert datasets[name][1] == field_units


@pytest.mark.parametrize(
    ("refusal", "error", "message"),
    [
        ("record_s", ValueError, "recording period must be positive"),
        ("duration_s", ValueError, "duration must be positive"),
        ("absent field", ValueError, "'bad': its message has no field 'r_BN_N'"),
        ("already run", RuntimeError, "already run"),
    ],
)
def test_a_refused_run_leaves_the_scenario_unchanged(refusal, error, message):
    """A refused run adds nothing to the task, and a scenario that has not run still runs."""
    vehicle = ImuSpacecraft()
    scenario = Leo500(vehicle)
    arguments = {"duration_s": 10.0, "record_s": 10.0}
    if refusal in arguments:
        arguments[refusal] = 0.0
    elif refusal == "absent field":
        fields = {**vehicle.recorded_fields(), "bad": (vehicle.imu.sensorOutMsg, "r_BN_N", "m")}
        vehicle.recorded_fields = lambda: fields
    else:
        scenario.run(inertial3D.inertial3D(), duration_s=1.0)
    models_before = len(scenario._simulation.TaskList[0].TaskModels)

    with pytest.raises(error, match=message):
        scenario.run(inertial3D.inertial3D(), **arguments)
    assert len(scenario._simulation.TaskList[0].TaskModels) == models_before
    if refusal in ("record_s", "duration_s"):
        scenario.run(inertial3D.inertial3D(), duration_s=10.0)
