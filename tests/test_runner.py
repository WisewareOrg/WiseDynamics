"""`run_case` and `Recording`: what a case records and how it becomes run-file datasets."""

import numpy as np
import pytest
from Basilisk.fswAlgorithms import inertial3D

from tests.support import StubFsw, run_with_stub
from wise_dynamics.basilisk import ImuSpacecraft, RecordedField, run_case


def test_recording_holds_truth_and_flight_software_roles_on_the_record_cadence():
    recording = run_with_stub(ImuSpacecraft(), duration_s=50.0, record_s=10.0)
    assert set(recording) == {"truth", "imu", "stub"}

    datasets = recording.datasets(
        {
            **ImuSpacecraft.recorded_fields,
            "est/sigma_BN": RecordedField("stub", "sigma_BN", "MRP"),
        }
    )
    t_s, units = datasets["t_s"]
    assert units == "s"
    assert np.array_equal(t_s, np.arange(0.0, 50.0 + 10.0, 10.0))
    assert list(datasets) == ["t_s", *ImuSpacecraft.recorded_fields, "est/sigma_BN"]
    for name, field in ImuSpacecraft.recorded_fields.items():
        assert datasets[name].data.shape == (6, 3)
        assert datasets[name].units == field.units
    assert np.allclose(datasets["est/sigma_BN"].data[-1], StubFsw.FIXED_SIGMA_BN)


def test_run_case_runs_a_stock_basilisk_module():
    module = inertial3D.inertial3D()
    module.ModelTag = "inertial3D"
    module.sigma_R0N = [0.1, 0.0, 0.0]
    recording = run_case(
        ImuSpacecraft(), module, fsw_outputs={"att_ref": module.attRefOutMsg}, duration_s=20.0
    )
    sigma_RN = recording.datasets({"ref": RecordedField("att_ref", "sigma_RN", "MRP")})["ref"]
    assert np.allclose(sigma_RN.data[-1], [0.1, 0.0, 0.0])


@pytest.mark.parametrize(
    ("refusal", "error", "message"),
    [
        ("truth role", ValueError, r"truth roles \['truth'\]"),
        ("record_s", ValueError, "recording period must be positive"),
        ("duration_s", ValueError, "duration must be positive"),
        ("already run", RuntimeError, "already run"),
    ],
)
def test_a_refused_call_leaves_the_plant_unchanged(refusal, error, message):
    plant = ImuSpacecraft()
    stub = StubFsw()
    arguments = {"fsw_outputs": {"stub": stub.output_msg}, "duration_s": 10.0, "record_s": 10.0}
    if refusal == "truth role":
        arguments["fsw_outputs"] = {"truth": stub.output_msg}
    elif refusal in ("record_s", "duration_s"):
        arguments[refusal] = 0.0
    else:
        plant.execute(1.0)
    models_before = len(plant.simulation.TaskList[0].TaskModels)

    with pytest.raises(error, match=message):
        run_case(plant, stub, **arguments)
    assert len(plant.simulation.TaskList[0].TaskModels) == models_before


def test_a_field_absent_from_its_recorder_is_named():
    recording = run_with_stub(ImuSpacecraft(), duration_s=10.0)
    with pytest.raises(ValueError, match="r_BN_N"):
        recording.datasets({"bad": RecordedField("imu", "r_BN_N", "m")})
