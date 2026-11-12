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
    ("fidelity", "integrator_type"),
    [
        ("hi", svIntegrators.svIntegratorRKF78),
        ("med", svIntegrators.svIntegratorRK4),
        ("lo", svIntegrators.svIntegratorRungeKutta3),
    ],
)
def test_each_fidelity_gives_its_integrator_to_the_vehicle_that_runs(
    fidelity: str, integrator_type: type
):
    """Each fidelity gives the vehicle that runs its integrator; `hi` states its tolerances.

    Args:
        fidelity: The fidelity level set on the scenario.
        integrator_type: The Basilisk integrator class the level gives.
    """
    scenario = Leo500()
    scenario.fidelity = fidelity
    scenario.disperse(np.random.default_rng(0), noise_seed=1)
    scenario.run(_Guidance(), {}, duration_s=1.0)
    (integrator,) = scenario._integrators
    assert type(integrator) is integrator_type
    assert int(scenario.vehicle.spacecraft.integrator.this) == int(integrator.this)
    if fidelity == "hi":
        assert integrator.getRelativeTolerance() == 1e-10
        assert integrator.getAbsoluteTolerance() == 1e-12


@pytest.mark.parametrize(
    ("period_s", "record_s", "step_ns"),
    [
        (1.0, 10.0, 1_000_000_000),
        (10.0, 10.0, 1_000_000_000),
        (0.1, 10.0, 100_000_000),
        (0.25, 10.0, 250_000_000),
        (1.0, 0.5, 500_000_000),
        (2.5, 10.0, 625_000_000),
    ],
)
def test_the_step_is_the_largest_of_at_most_1_s_dividing_the_fsw_and_record_periods(
    period_s: float, record_s: float, step_ns: int
):
    """The dynamics steps at the largest step of at most 1 s that divides both periods.

    The sensors and the flight software then run at the flight software's period, each on an
    instant the dynamics has just reached.

    Args:
        period_s: The flight software's period, in seconds.
        record_s: The recording period, in seconds.
        step_ns: The dynamics task period the two give, in nanoseconds.
    """
    scenario = Leo500()
    scenario.record_s = record_s
    fsw = type("_Paced", (_Guidance,), {"period_s": period_s})()
    scenario.run(fsw, {}, duration_s=0.0)
    dynamics, sensors, flight, record = scenario._simulation.TaskList
    assert dynamics.TaskData.TaskPeriod == step_ns
    assert sensors.TaskData.TaskPeriod == round(period_s * 1e9)
    assert flight.TaskData.TaskPeriod == round(period_s * 1e9)
    assert record.TaskData.TaskPeriod == round(record_s * 1e9)


def test_the_dynamics_task_holds_the_models_and_the_sensor_task_the_sensors():
    """The propagated modules step with the dynamics; the sensors only at the FSW period."""
    scenario = Leo500()
    vehicle = scenario.vehicle
    scenario.run(_Guidance(), {}, duration_s=1.0)
    dynamics, sensors, _, _ = scenario._simulation.TaskList
    assert dynamics.TaskModels[: len(vehicle.models())] == vehicle.models()
    assert sensors.TaskModels == [vehicle.imu]


def test_two_fidelities_of_one_case_draw_the_same_sensor_noise():
    """Sampled at the flight software's period, a case's IMU noise does not depend on the tier.

    What differs between the tiers' IMU outputs is the numerical difference in the truth they
    measure, far below the noise.
    """
    gyro = {}
    for fidelity in ("lo", "hi"):
        scenario = Leo500()
        scenario.fidelity = fidelity
        scenario.disperse(np.random.default_rng(3), noise_seed=5)
        gyro[fidelity] = scenario.run(_Guidance(), {}, duration_s=600.0)["imu/gyro_P"][0]
    assert np.abs(gyro["lo"] - gyro["hi"]).max() < 1e-9


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


def test_the_sensors_sample_at_the_flight_software_period_not_the_step():
    """At a 2 s flight-software period and 1 s recording, the dynamics steps 1 s and the IMU 2 s.

    The flight software reads the IMU output written at its own instant, and the IMU output
    recorded at each odd second is the one written a second before.
    """
    scenario = Leo500()
    scenario.record_s = 1.0
    paced = type("_Paced", (_Stepping,), {"period_s": 2.0})
    fsw = paced(scenario.vehicle.readers()["imu"])
    datasets = scenario.run(fsw, {}, duration_s=6.0)
    dynamics, *_ = scenario._simulation.TaskList
    assert dynamics.TaskData.TaskPeriod == 1_000_000_000
    assert fsw.called_ns == [k * 2_000_000_000 for k in range(4)]
    assert fsw.read_ns == fsw.called_ns
    gyro = datasets["imu/gyro_P"][0]
    assert np.array_equal(gyro[1::2], gyro[0:-1:2])
    assert not np.array_equal(gyro[2::2], gyro[1::2])
