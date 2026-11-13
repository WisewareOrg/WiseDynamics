"""`Scenario.run`: what a run records, as run-file datasets, and that it runs once."""

import numpy as np
import pytest
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.simulation import svIntegrators

from wise_dynamics.basilisk import ImuSpacecraft
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


def test_a_run_records_the_vehicle_and_fsw_fields_at_every_dynamics_step():
    """A run records `t_s`, the vehicle's fields and the flight-software fields at every step."""
    scenario = Leo500()
    fields = scenario.vehicle.recorded_fields()
    guidance, fsw_fields = _guidance()
    datasets = scenario.run(guidance, fsw_fields, duration_s=50.0)
    t_s, units = datasets["t_s"]
    assert units == "s"
    assert np.array_equal(t_s, np.arange(0.0, 51.0, 1.0))
    assert list(datasets) == ["t_s", *fields, "guidance/sigma_RN"]
    for name, (_, _, field_units) in fields.items():
        assert len(datasets[name][0]) == 51
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
    ("period_s", "step_ns"),
    [
        (1.0, 1_000_000_000),
        (10.0, 1_000_000_000),
        (0.1, 100_000_000),
        (0.25, 250_000_000),
        (2.5, 625_000_000),
        (1 / 3, 333_333_333),
    ],
)
def test_the_step_is_the_largest_of_at_most_1_s_dividing_the_fsw_period(
    period_s: float, step_ns: int
):
    """The dynamics steps at the largest step of at most 1 s that divides the FSW's period.

    The sensors and the flight software then run at the flight software's period, and the
    recorders at every step, each on an instant the dynamics has just reached.

    Args:
        period_s: The flight software's period, in seconds.
        step_ns: The dynamics task period the flight software's period gives, in nanoseconds.
    """
    scenario = Leo500()
    fsw = type("_Paced", (_Guidance,), {"period_s": period_s})()
    scenario.run(fsw, {}, duration_s=0.0)
    dynamics, sensors, flight, record = scenario._simulation.TaskList
    assert dynamics.TaskData.TaskPeriod == step_ns
    assert sensors.TaskData.TaskPeriod == round(period_s * 1e9)
    assert flight.TaskData.TaskPeriod == round(period_s * 1e9)
    assert record.TaskData.TaskPeriod == step_ns


def test_a_flight_software_period_with_no_step_of_at_least_1_ms_is_refused():
    """A period of 1000000007 ns, a prime, divides into no step between 1 ms and 1 s."""
    scenario = Leo500()
    fsw = type("_Paced", (_Guidance,), {"period_s": 1.000000007})()
    with pytest.raises(ValueError, match=r"the 1000000007 ns flight-software period"):
        scenario.run(fsw, {}, duration_s=0.0)


def test_the_dynamics_task_holds_the_models_and_the_sensor_task_the_sensors():
    """The propagated modules step with the dynamics; the sensors only at the FSW period."""
    scenario = Leo500()
    vehicle = scenario.vehicle
    scenario.run(_Guidance(), {}, duration_s=1.0)
    dynamics, sensors, _, _ = scenario._simulation.TaskList
    assert dynamics.TaskModels == vehicle.models()
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
    assert np.std(gyro["hi"], axis=0).min() > 1e-6
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
    """The module runs at whole seconds only, and is recorded from its own instant at every step.

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
    assert t_s.tolist() == [float(k) for k in range(21)]
    assert np.array_equal(datasets["fsw/time_tag"][0], t_s + 100.0)


def test_the_sensors_sample_at_the_flight_software_period_not_the_step():
    """At a 2 s flight-software period and 1 s recording, the dynamics steps 1 s and the IMU 2 s.

    The flight software reads the IMU output written at its own instant, and the IMU output
    recorded at each odd second is the one written a second before.
    """
    scenario = Leo500()
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


class _Marker(sysModel.SysModel):
    """A module that notes its name and call time in a list shared with other modules."""

    def __init__(self, name: str, order: list[tuple[int, str]], period_s: float = 1.0) -> None:
        """Hold the name, the shared list and the period.

        Args:
            name: What the module notes.
            order: The list every module notes `(time in ns, name)` into.
            period_s: The module's period, in seconds.
        """
        super().__init__()
        self.name, self.order, self.period_s = name, order, period_s

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Note the call.

        Args:
            CurrentSimNanos: The simulation time of this call, in nanoseconds.
        """
        self.order.append((CurrentSimNanos, self.name))


def test_the_ground_runs_after_the_flight_software_at_every_shared_instant():
    """The ground, in its own process, is called after the flight software at each instant."""
    order: list[tuple[int, str]] = []
    scenario = Leo500()
    scenario.run(
        _Marker("fsw", order),
        {},
        duration_s=3.0,
        ground=_Marker("ground", order),
    )
    assert order == [(k * 1_000_000_000, name) for k in range(4) for name in ("fsw", "ground")]


def test_the_ground_has_a_task_of_its_own_at_its_own_period():
    """The ground's task steps at the ground's period, in a process separate from the scenario's."""
    scenario = Leo500()
    scenario.run(_Marker("fsw", []), {}, duration_s=0.0, ground=_Marker("ground", [], period_s=2.0))
    *_, ground = scenario._simulation.TaskList
    assert ground.TaskData.TaskPeriod == 2_000_000_000
    assert [process.Name for process in scenario._simulation.procList] == [
        "scenario_process",
        "ground_process",
    ]


class _Stopping(_Marker):
    """A ground that stops its own task at its call at a given time."""

    def __init__(self, order: list[tuple[int, str]], scenario: Leo500, at_ns: int) -> None:
        """Hold the shared list, the scenario that stops it and when.

        Args:
            order: The list every module notes `(time in ns, name)` into.
            scenario: The scenario running it.
            at_ns: The time of the call that stops it, in nanoseconds.
        """
        super().__init__("ground", order)
        self.scenario, self.at_ns = scenario, at_ns

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Note the call, and stop the ground's task at `at_ns`.

        Args:
            CurrentSimNanos: The simulation time of this call, in nanoseconds.
        """
        super().UpdateState(CurrentSimNanos)
        if CurrentSimNanos == self.at_ns:
            self.scenario.stop_ground()


def test_a_ground_that_stops_is_not_called_again_while_the_flight_software_runs_on():
    """After `stop_ground` at t = 1 s, the ground is never called again; the run goes on."""
    order: list[tuple[int, str]] = []
    scenario = Leo500()
    ground = _Stopping(order, scenario, at_ns=1_000_000_000)
    scenario.run(_Marker("fsw", order), {}, duration_s=3.0, ground=ground)
    assert [t for t, name in order if name == "ground"] == [0, 1_000_000_000]
    assert [t for t, name in order if name == "fsw"] == [k * 1_000_000_000 for k in range(4)]


class _Exact(ImuSpacecraft):
    """An `ImuSpacecraft` without biases or noise, so its gyro reads the body rate it samples.

    Attributes:
        gyro_bias_P_rad_s: Constant gyro bias in the IMU platform frame, in rad/s.
        accel_bias_P_m_s2: Constant accelerometer bias in the IMU platform frame, in m/s^2.
        gyro_noise_density_rad_s_rthz: Gyro white-noise density, in rad/s/sqrt(Hz).
        accel_noise_density_m_s2_rthz: Accelerometer white-noise density, in m/s^2/sqrt(Hz).
    """

    gyro_bias_P_rad_s = (0.0, 0.0, 0.0)
    accel_bias_P_m_s2 = (0.0, 0.0, 0.0)
    gyro_noise_density_rad_s_rthz = 0.0
    accel_noise_density_m_s2_rthz = 0.0


def test_the_sensors_sample_the_state_the_dynamics_has_just_reached():
    """At each instant the dynamics steps first, so the gyro reads that instant's body rate.

    The tumbling body's rate changes every step, so a sensor sampling before the dynamics would
    read the previous step's rate. Basilisk's `ImuSensor` writes zeros on its first call, at t = 0,
    so the comparison starts at the next sample.
    """
    scenario = Leo500(vehicle_type=_Exact)
    datasets = scenario.run(_Guidance(), {}, duration_s=60.0)
    omega = datasets["truth/omega_BN_B"][0]
    assert np.abs(np.diff(omega, axis=0)).max() > 1e-6
    assert np.array_equal(datasets["imu/gyro_P"][0][1:], omega[1:])
