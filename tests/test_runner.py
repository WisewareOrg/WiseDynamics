"""`run_case` and `Recording`: what a case records and how it becomes run-file datasets."""

import gc
import weakref

import numpy as np
import pytest
from Basilisk.architecture import messaging, sysModel
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


def churn_plants() -> list[ImuSpacecraft]:
    """New plants allocated after a plant is freed, reusing its memory."""
    gc.collect()
    return [ImuSpacecraft() for _ in range(10)]


def test_a_recording_outlives_its_plant():
    plant = ImuSpacecraft()
    released = weakref.ref(plant)
    recording = run_with_stub(plant, duration_s=50.0)
    before = recording.datasets(plant.recorded_fields)
    fields = dict(plant.recorded_fields)
    del plant
    others = churn_plants()

    assert released() is None, "the plant is still referenced"
    after = recording.datasets(fields)
    assert others
    for name, dataset in before.items():
        assert np.array_equal(after[name].data, dataset.data), name


class ReadingFsw(sysModel.SysModel):
    """A flight-software module holding the plant's IMU reader, as a hosted algorithm does."""

    def __init__(self, reader) -> None:
        super().__init__()
        self.ModelTag = "reading_fsw"
        self.reader = reader
        self.output_msg = messaging.NavAttMsg()

    def UpdateState(self, CurrentSimNanos: int) -> None:
        payload = messaging.NavAttMsgPayload()
        payload.omega_BN_B = self.reader().AngVelPlatform
        self.output_msg.write(payload, CurrentSimNanos, self.moduleID)


def test_a_plant_is_freed_after_its_case():
    """A case whose flight software holds the plant's reader leaves nothing referencing the
    plant once the case's own references go, so a session's plants do not accumulate."""
    plant = ImuSpacecraft()
    released = weakref.ref(plant)
    fsw = ReadingFsw(plant.readers()["imu"])
    run_case(plant, fsw, fsw_outputs={"fsw": fsw.output_msg}, duration_s=20.0)
    del plant, fsw
    gc.collect()
    assert released() is None, "the plant is still referenced after its case"
