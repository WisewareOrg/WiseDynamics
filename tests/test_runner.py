"""`run_case`: what a case records, as run-file datasets, and what it refuses."""

import gc
import weakref

import numpy as np
import pytest
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import inertial3D

from tests.support import GUIDANCE_SIGMA_RN, run_with_guidance
from wise_dynamics.basilisk import ImuSpacecraft, run_case


def test_a_case_records_its_fields_on_the_record_cadence():
    plant = ImuSpacecraft()
    fields = plant.recorded_fields()
    datasets = run_with_guidance(plant, duration_s=50.0, record_s=10.0)
    t_s, units = datasets["t_s"]
    assert units == "s"
    assert np.array_equal(t_s, np.arange(0.0, 50.0 + 10.0, 10.0))
    assert list(datasets) == ["t_s", *fields, "guidance/sigma_RN"]
    for name, (_, _, field_units) in fields.items():
        assert datasets[name][0].shape == (6, 3)
        assert datasets[name][1] == field_units
    assert np.allclose(datasets["guidance/sigma_RN"][0][-1], GUIDANCE_SIGMA_RN)


@pytest.mark.parametrize(
    ("refusal", "error", "message"),
    [
        ("no fields", ValueError, "at least one field to record"),
        ("record_s", ValueError, "recording period must be positive"),
        ("duration_s", ValueError, "duration must be positive"),
        ("already run", RuntimeError, "already run"),
    ],
)
def test_a_refused_call_leaves_the_plant_unchanged(refusal, error, message):
    plant = ImuSpacecraft()
    guidance = inertial3D.inertial3D()
    arguments = {"fields": plant.recorded_fields(), "duration_s": 10.0, "record_s": 10.0}
    if refusal == "no fields":
        arguments["fields"] = {}
    elif refusal in ("record_s", "duration_s"):
        arguments[refusal] = 0.0
    else:
        plant.execute(1.0)
    models_before = len(plant._simulation.TaskList[0].TaskModels)

    with pytest.raises(error, match=message):
        run_case(plant, guidance, **arguments)
    assert len(plant._simulation.TaskList[0].TaskModels) == models_before


def test_a_field_absent_from_its_message_is_named():
    plant = ImuSpacecraft()
    with pytest.raises(ValueError, match="r_BN_N"):
        run_case(
            plant,
            inertial3D.inertial3D(),
            fields={"bad": (plant.imu.sensorOutMsg, "r_BN_N", "m")},
            duration_s=10.0,
        )


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
    run_case(plant, fsw, fields=plant.recorded_fields(), duration_s=20.0)
    del plant, fsw
    gc.collect()
    assert released() is None, "the plant is still referenced after its case"
