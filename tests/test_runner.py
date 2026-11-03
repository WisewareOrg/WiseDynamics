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
    vehicle = ImuSpacecraft()
    fields = vehicle.recorded_fields()
    datasets = run_with_guidance(vehicle, duration_s=50.0, record_s=10.0)
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
        ("absent field", ValueError, "'bad': its message has no field 'r_BN_N'"),
        ("record_s", ValueError, "recording period must be positive"),
        ("duration_s", ValueError, "duration must be positive"),
        ("already run", RuntimeError, "already run"),
    ],
)
def test_a_refused_call_leaves_the_vehicle_unchanged(refusal, error, message):
    """A refused call adds nothing to the vehicle, and a vehicle that has not run still runs."""
    vehicle = ImuSpacecraft()
    arguments = {"fields": vehicle.recorded_fields(), "duration_s": 10.0, "record_s": 10.0}
    if refusal == "no fields":
        arguments["fields"] = {}
    elif refusal == "absent field":
        arguments["fields"] = {
            **arguments["fields"],
            "bad": (vehicle.imu.sensorOutMsg, "r_BN_N", "m"),
        }
    elif refusal in ("record_s", "duration_s"):
        arguments[refusal] = 0.0
    else:
        vehicle.execute(1.0)
    models_before = len(vehicle._simulation.TaskList[0].TaskModels)

    with pytest.raises(error, match=message):
        run_case(vehicle, inertial3D.inertial3D(), **arguments)
    assert len(vehicle._simulation.TaskList[0].TaskModels) == models_before
    if refusal != "already run":
        run_case(
            vehicle, inertial3D.inertial3D(), fields=vehicle.recorded_fields(), duration_s=10.0
        )


class ReadingFsw(sysModel.SysModel):
    """A flight-software module holding the vehicle's IMU reader, as a hosted algorithm does."""

    def __init__(self, reader) -> None:
        super().__init__()
        self.ModelTag = "reading_fsw"
        self.reader = reader
        self.output_msg = messaging.NavAttMsg()

    def UpdateState(self, CurrentSimNanos: int) -> None:
        payload = messaging.NavAttMsgPayload()
        payload.omega_BN_B = self.reader().AngVelPlatform
        self.output_msg.write(payload, CurrentSimNanos, self.moduleID)


def test_a_vehicle_is_freed_after_its_case():
    """A case whose flight software holds the vehicle's reader leaves nothing referencing the
    vehicle once the case's own references go, so a session's vehicles do not accumulate."""
    vehicle = ImuSpacecraft()
    released = weakref.ref(vehicle)
    fsw = ReadingFsw(vehicle.readers()["imu"])
    run_case(vehicle, fsw, fields=vehicle.recorded_fields(), duration_s=20.0)
    del vehicle, fsw
    gc.collect()
    assert released() is None, "the vehicle is still referenced after its case"
