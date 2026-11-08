"""`Scenario.run`: what a run records, as run-file datasets, and that it runs once."""

import numpy as np
import pytest
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.simulation import svIntegrators

from wise_dynamics.basilisk.scenarios.leo500 import Leo500

GUIDANCE_SIGMA_RN = [0.1, 0.2, 0.3]


class _Guidance(inertial3D.inertial3D):
    """Basilisk's stock `inertial3D` standing in for flight software, stepping at 1 Hz.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 1.0


def _guidance():
    """Build Basilisk's stock `inertial3D` standing in for flight software.

    Returns:
        The guidance module, holding `GUIDANCE_SIGMA_RN`, and its `guidance/sigma_RN` field to
        record, as `fsw_fields`.
    """
    guidance = _Guidance()
    guidance.sigma_R0N = GUIDANCE_SIGMA_RN
    return guidance, {"guidance/sigma_RN": (guidance.attRefOutMsg, "sigma_RN", "MRP")}


def test_a_run_records_the_vehicle_and_fsw_fields_on_the_record_cadence():
    """A run records `t_s`, the vehicle's fields and the flight-software fields every `record_s`."""
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


@pytest.mark.parametrize(
    ("fidelity", "step_ns", "integrator_type"),
    [
        ("hi", 20_000_000, svIntegrators.svIntegratorRK4),
        ("med", 50_000_000, svIntegrators.svIntegratorRK4),
        ("lo", 50_000_000, svIntegrators.svIntegratorRK2),
    ],
)
def test_each_fidelity_sets_its_step_and_integrator_on_the_vehicle_that_runs(
    fidelity: str, step_ns: int, integrator_type: type
):
    """Each fidelity sets the task step and the integrator of the vehicle that runs.

    Args:
        fidelity: The fidelity level set on the scenario.
        step_ns: The dynamics task period the level gives, in nanoseconds.
        integrator_type: The Basilisk integrator class the level gives.
    """
    scenario = Leo500()
    scenario.fidelity = fidelity
    scenario.disperse(np.random.default_rng(0), noise_seed=1)
    scenario.run(_Guidance(), {}, duration_s=1.0)
    dynamics, fsw, record = scenario._simulation.TaskList
    assert dynamics.TaskData.TaskPeriod == step_ns
    assert fsw.TaskData.TaskPeriod == 1_000_000_000
    assert record.TaskData.TaskPeriod == 10_000_000_000
    assert dynamics.TaskModels[:2] == scenario.vehicle.models()
    (integrator,) = scenario._integrators
    assert type(integrator) is integrator_type
    assert int(scenario.vehicle.spacecraft.integrator.this) == int(integrator.this)


def test_a_scenario_runs_once():
    """A second `run` of the same scenario is refused."""
    scenario = Leo500()
    scenario.run(_Guidance(), {}, duration_s=1.0)
    with pytest.raises(RuntimeError, match="already run"):
        scenario.run(_Guidance(), {}, duration_s=1.0)


class _Stepping(sysModel.SysModel):
    """A flight-software module that notes its calls and writes its call time plus 100 s.

    It also notes when the IMU output it reads was written.

    Attributes:
        period_s: The module's period, in seconds.
    """

    period_s = 1.0

    def __init__(self, reader: messaging.IMUSensorMsgReader) -> None:
        """Hold the IMU reader and create the module's output message.

        Args:
            reader: The vehicle's IMU message reader.
        """
        super().__init__()
        self.ModelTag = "stepping"
        self.reader = reader
        self.called_ns: list[int] = []
        self.read_ns: list[int] = []
        self.output_msg = messaging.NavAttMsg()

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Note the call time and the IMU output's write time, and write the call time plus 100 s.

        Args:
            CurrentSimNanos: The simulation time of this call, in nanoseconds.
        """
        self.called_ns.append(CurrentSimNanos)
        self.read_ns.append(self.reader.timeWritten())
        payload = messaging.NavAttMsgPayload()
        payload.timeTag = CurrentSimNanos / 1e9 + 100.0
        self.output_msg.write(payload, CurrentSimNanos, self.moduleID)


def test_the_flight_software_steps_at_its_own_period_and_is_recorded_at_its_instant():
    """The module runs at whole seconds only, and is recorded from its own instant.

    It reads the IMU output of that instant, and each recorded value is the one it wrote at that
    row's own time, t = 0 included.
    """
    scenario = Leo500()
    fsw = _Stepping(scenario.vehicle.readers()["imu"])
    datasets = scenario.run(
        fsw, {"fsw/time_tag": (fsw.output_msg, "timeTag", "s")}, duration_s=20.0
    )
    assert fsw.called_ns == [k * 1_000_000_000 for k in range(21)]
    assert fsw.read_ns == fsw.called_ns
    t_s, _ = datasets["t_s"]
    assert t_s.tolist() == [0.0, 10.0, 20.0]
    assert np.array_equal(datasets["fsw/time_tag"][0], t_s + 100.0)
