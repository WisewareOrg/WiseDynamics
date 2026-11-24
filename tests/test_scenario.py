"""`Scenario.run`: what a run records, as run-file datasets, and that it runs once."""

import contextvars
import ctypes
import threading
import weakref
from pathlib import Path

import numpy as np
import pytest
from Basilisk.architecture import messaging, sysModel
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.simulation import svIntegrators

from tests.allocations import assert_no_native_allocations_per_tick
from wise_dynamics.basilisk import FswHost, ImuSpacecraft
from wise_dynamics.basilisk import scenario as scenario_module
from wise_dynamics.basilisk._thread_state import ThreadStateHold
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
    scenario = Leo500(ImuSpacecraft())
    fields = scenario.vehicle.recorded_fields()
    guidance, fsw_fields = _guidance()
    datasets = scenario.run(guidance, fsw_fields, duration_s=50.0)
    t_s, units = datasets["t_s"]
    assert units == "s"
    assert np.array_equal(t_s, np.arange(0.0, 51.0, 1.0))
    assert list(datasets) == ["t_s", *fields, "guidance/sigma_RN"]
    widths = {"truth/q_BN": 4, "guidance/sigma_RN": 3}
    for name, (_, _, field_units) in fields.items():
        assert datasets[name][0].shape == (51, widths.get(name, 3)), name
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
def test_each_fidelity_gives_its_integrator_to_the_vehicle_that_runs(fidelity: str, integrator_type: type):
    """Each fidelity gives the vehicle that runs its integrator; `hi` states its tolerances.

    Args:
        fidelity: The fidelity level set on the scenario.
        integrator_type: The Basilisk integrator class the level gives.
    """
    scenario = Leo500(ImuSpacecraft())
    scenario.fidelity = fidelity
    scenario.disperse_case(np.random.SeedSequence(0))
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
def test_the_step_is_the_largest_of_at_most_1_s_dividing_the_fsw_period(period_s: float, step_ns: int):
    """The dynamics steps at the largest step of at most 1 s that divides the FSW's period.

    The sensors and the flight software then run at the flight software's period, and the
    recorders at every step, each on an instant the dynamics has just reached.

    Args:
        period_s: The flight software's period, in seconds.
        step_ns: The dynamics task period the flight software's period gives, in nanoseconds.
    """
    scenario = Leo500(ImuSpacecraft())
    fsw = type("_Paced", (_Guidance,), {"period_s": period_s})()
    simulation = scenario._simulation
    scenario.run(fsw, {}, duration_s=0.0)
    dynamics, sensors, flight, record = simulation.TaskList
    assert dynamics.TaskData.TaskPeriod == step_ns
    assert sensors.TaskData.TaskPeriod == round(period_s * 1e9)
    assert flight.TaskData.TaskPeriod == round(period_s * 1e9)
    assert record.TaskData.TaskPeriod == step_ns


def test_a_flight_software_period_with_no_step_of_at_least_1_ms_is_refused():
    """A period of 1000000007 ns, a prime, divides into no step between 1 ms and 1 s."""
    scenario = Leo500(ImuSpacecraft())
    fsw = type("_Paced", (_Guidance,), {"period_s": 1.000000007})()
    with pytest.raises(ValueError, match=r"the 1000000007 ns flight-software period"):
        scenario.run(fsw, {}, duration_s=0.0)


def test_the_dynamics_task_holds_the_models_and_the_sensor_task_the_sensors():
    """The propagated modules step with the dynamics; the sensors only at the FSW period."""
    vehicle = ImuSpacecraft()
    scenario = Leo500(vehicle)
    simulation = scenario._simulation
    scenario.run(_Guidance(), {}, duration_s=1.0)
    tasks = {task.Name: task for task in simulation.TaskList}
    dynamics, sensors = tasks["dynamics_task"], tasks["sensor_task"]
    assert dynamics.TaskModels == vehicle.models()
    assert sensors.TaskModels == [vehicle.imu]


def test_two_fidelities_of_one_case_draw_the_same_sensor_noise():
    """Sampled at the flight software's period, a case's IMU noise does not depend on the tier.

    The tiers' IMU outputs differ by exactly their truth difference: the bias and the noise are
    identical.
    """
    gyro = {}
    truth = {}
    for fidelity in ("lo", "hi"):
        scenario = Leo500(ImuSpacecraft())
        scenario.fidelity = fidelity
        scenario.disperse_case(np.random.SeedSequence(3))
        datasets = scenario.run(_Guidance(), {}, duration_s=600.0)
        gyro[fidelity] = datasets["imu/gyro_P"][0]
        truth[fidelity] = datasets["truth/omega_BN_B"][0]
    assert np.std(gyro["hi"], axis=0).min() > 1e-6
    noise_difference = (gyro["lo"] - gyro["hi"]) - (truth["lo"] - truth["hi"])
    assert np.abs(noise_difference).max() < 1e-15, (
        f"the tiers' noise differs by {np.abs(noise_difference).max():.2e} rad/s; expected equal to rounding, < 1e-15"
    )


def test_a_scenario_runs_once():
    """A second `run` of the same scenario is refused."""
    scenario = Leo500(ImuSpacecraft())
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
    scenario = Leo500(ImuSpacecraft())
    fsw = _Stepping(scenario.vehicle.readers()["imu"])
    datasets = scenario.run(fsw, {"fsw/time_tag": (fsw.output_msg, "timeTag", "s")}, duration_s=20.0)
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
    scenario = Leo500(ImuSpacecraft())
    paced = type("_Paced", (_Stepping,), {"period_s": 2.0})
    fsw = paced(scenario.vehicle.readers()["imu"])
    simulation = scenario._simulation
    datasets = scenario.run(fsw, {}, duration_s=6.0)
    dynamics, *_ = simulation.TaskList
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
    scenario = Leo500(ImuSpacecraft())
    scenario.run(
        _Marker("fsw", order),
        {},
        duration_s=3.0,
        ground=_Marker("ground", order),
    )
    assert order == [(k * 1_000_000_000, name) for k in range(4) for name in ("fsw", "ground")]


def test_the_ground_runs_at_its_own_period_in_a_process_of_its_own():
    """A 2 s ground is called every 2 s while the 1 s flight software runs every second."""
    order: list[tuple[int, str]] = []
    scenario = Leo500(ImuSpacecraft())
    ground = _Marker("ground", order, period_s=2.0)
    simulation = scenario._simulation
    scenario.run(_Marker("fsw", order), {}, duration_s=4.0, ground=ground)
    assert [t for t, name in order if name == "ground"] == [k * 2_000_000_000 for k in range(3)]
    assert [t for t, name in order if name == "fsw"] == [k * 1_000_000_000 for k in range(5)]
    assert [process.Name for process in simulation.procList] == [
        "scenario_process",
        "ground_process",
        "thread_state_process",
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
    scenario = Leo500(ImuSpacecraft())
    ground = _Stopping(order, scenario, at_ns=1_000_000_000)
    scenario.run(_Marker("fsw", order), {}, duration_s=3.0, ground=ground)
    assert [t for t, name in order if name == "ground"] == [0, 1_000_000_000]
    assert [t for t, name in order if name == "fsw"] == [k * 1_000_000_000 for k in range(4)]


class _StoppingThenRaising(_Stopping):
    """A ground that stops its own task at its call at a given time, then raises."""

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Stop the ground's task at `at_ns`, then raise there.

        Args:
            CurrentSimNanos: The simulation time of this call, in nanoseconds.

        Raises:
            RuntimeError: At `at_ns`.
        """
        super().UpdateState(CurrentSimNanos)
        if CurrentSimNanos == self.at_ns:
            raise RuntimeError("the ground fails")


def test_a_scenario_keeps_no_ground_once_its_run_returns():
    """With the scenario alive, a ground that holds the scenario is freed when `run` returns."""
    order: list[tuple[int, str]] = []
    scenario = Leo500(ImuSpacecraft())
    ground = _Stopping(order, scenario, at_ns=1_000_000_000)
    kept = weakref.ref(ground)

    scenario.run(_Marker("fsw", order), {}, duration_s=3.0, ground=ground)
    del ground

    assert kept() is None, "the scenario still reaches its ground after `run` returned"


def test_a_scenario_keeps_no_ground_once_its_run_raises():
    """With the scenario alive, a ground that holds the scenario is freed when `run` raises."""
    order: list[tuple[int, str]] = []
    scenario = Leo500(ImuSpacecraft())
    ground = _StoppingThenRaising(order, scenario, at_ns=1_000_000_000)
    kept = weakref.ref(ground)

    with pytest.raises(RuntimeError, match="SWIG director method error"):
        scenario.run(_Marker("fsw", order), {}, duration_s=3.0, ground=ground)
    del ground

    assert kept() is None, "the scenario still reaches its ground after `run` raised"


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
    scenario = Leo500(_Exact())
    datasets = scenario.run(_Guidance(), {}, duration_s=60.0)
    omega = datasets["truth/omega_BN_B"][0]
    assert np.abs(np.diff(omega, axis=0)).max() > 1e-6
    assert np.array_equal(datasets["imu/gyro_P"][0][1:], omega[1:])


def test_a_tick_of_the_host_makes_no_native_allocation(tmp_path: Path, commanding: type):
    """Basilisk's call into `FswHost` and back, with a build commanding nothing, allocates nothing per tick.

    Args:
        tmp_path: The test's own temporary directory.
        commanding: The stub build class.
    """

    def fly(ticks: int) -> int:
        """Fly `ticks` ticks of a build commanding nothing.

        Args:
            ticks: The number of ticks to fly.

        Returns:
            The ticks the build ran.
        """
        build = commanding({})
        scenario = Leo500(ImuSpacecraft())
        scenario.run(FswHost(build, scenario.vehicle), {}, duration_s=ticks * build.period_s)
        return build.ticked

    assert_no_native_allocations_per_tick(
        fly, within="SwigDirector__SysModel::UpdateState", ticks=100, directory=tmp_path
    )


class _Remembering(sysModel.SysModel):
    """A Python module that records, at each step, the time a thread-local and a context value were last set to.

    Attributes:
        period_s: The module's period, in seconds.
        seen: At each step, the time the thread-local value held, `None` while unset.
        seen_in_context: At each step, the time the context variable held, `None` while unset.
    """

    period_s = 1.0
    seen: list[int | None]
    seen_in_context: list[int | None]

    def __init__(self) -> None:
        """Start with nothing seen and an unset thread-local value."""
        super().__init__()
        self.seen = []
        self.seen_in_context = []
        self._local = threading.local()
        self._context: contextvars.ContextVar[int] = contextvars.ContextVar(f"context_{id(self)}")

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Record the thread-local and the context value, then set both to this step's time.

        Args:
            CurrentSimNanos: The simulation time, in nanoseconds.
        """
        self.seen.append(getattr(self._local, "t_ns", None))
        self.seen_in_context.append(self._context.get(None))
        self._local.t_ns = CurrentSimNanos
        self._context.set(CurrentSimNanos)


def test_the_python_modules_of_a_run_share_one_thread_state_for_the_run():
    """A thread-local or context value the flight software or the ground sets at one step is still set at its next.

    That holds at the final instant too, for the ground, the last module to run there.
    """
    fsw, ground = _Remembering(), _Remembering()

    Leo500(ImuSpacecraft()).run(fsw, {}, duration_s=3.0, ground=ground)

    steps = [None, 0, 1_000_000_000, 2_000_000_000]
    assert fsw.seen == steps, "the flight software's thread-local value did not last"
    assert ground.seen == steps, "the ground's thread-local value did not last"
    assert fsw.seen_in_context == steps, "the flight software's context value did not last"
    assert ground.seen_in_context == steps, "the ground's context value did not last"


@pytest.fixture
def flown_once() -> None:
    """Fly one run, so the threads a process's first run starts, such as tqdm's monitor, are running."""
    Leo500(ImuSpacecraft()).run(_Remembering(), {}, duration_s=1.0)


def _thread_states() -> int:
    """Count the interpreter's thread states, of live threads and of any left behind.

    Returns:
        The count.
    """
    api = ctypes.PyDLL(ctypes.pythonapi._name, handle=ctypes.pythonapi._handle)
    api.PyInterpreterState_Get.restype = ctypes.c_void_p
    api.PyInterpreterState_ThreadHead.restype = ctypes.c_void_p
    api.PyInterpreterState_ThreadHead.argtypes = [ctypes.c_void_p]
    api.PyThreadState_Next.restype = ctypes.c_void_p
    api.PyThreadState_Next.argtypes = [ctypes.c_void_p]
    count, state = 0, api.PyInterpreterState_ThreadHead(api.PyInterpreterState_Get())
    while state:
        count, state = count + 1, api.PyThreadState_Next(state)
    return count


@pytest.mark.usefixtures("flown_once")
def test_a_run_that_returns_leaves_no_thread_state_behind():
    """The thread state a run holds on Basilisk's thread is given back by the time the run returns."""
    before = _thread_states()

    Leo500(ImuSpacecraft()).run(_Remembering(), {}, duration_s=3.0)

    assert _thread_states() == before, "a run that returned left a thread state behind"


class _Held:
    """An object a failing call holds, whose life shows whether that call's error was let go."""


class _Raising(_Remembering):
    """A Python flight software that raises at its second step, holding a `_Held` as it does.

    Attributes:
        marker: A weak reference to the marker the raising call held.
    """

    marker: weakref.ref

    def UpdateState(self, CurrentSimNanos: int) -> None:
        """Raise at 1 s.

        Args:
            CurrentSimNanos: The simulation time, in nanoseconds.

        Raises:
            ValueError: At 1 s.
        """
        if CurrentSimNanos == 1_000_000_000:
            held = _Held()
            self.marker = weakref.ref(held)
            raise ValueError("raised by the flight software")


@pytest.mark.usefixtures("flown_once")
def test_a_run_that_raises_clears_the_thread_state_it_held_and_leaves_it_behind():
    """A run whose flight software raises keeps nothing of the failed call, and one cleared thread state.

    The thread state still held the failed call's error, and so its frame and the marker in it, until the
    run cleared it; only its own thread could delete it.
    """
    before = _thread_states()
    fsw = _Raising()

    with pytest.raises(RuntimeError, match="SWIG director method error"):
        Leo500(ImuSpacecraft()).run(fsw, {}, duration_s=3.0)

    assert fsw.marker() is None, "the failed call's frame is still alive"
    assert _thread_states() == before + 1, "a run that raised did not leave exactly one thread state behind"


class _RaisingAtReset(_Remembering):
    """A Python ground that raises at reset, before the run's thread state is taken."""

    def Reset(self, CurrentSimNanos: int) -> None:
        """Raise.

        Args:
            CurrentSimNanos: The reset time, in nanoseconds.

        Raises:
            ValueError: Always.
        """
        raise ValueError("raised by the ground at reset")


@pytest.mark.usefixtures("flown_once")
def test_a_run_whose_ground_raises_at_reset_leaves_no_thread_state_behind():
    """A module raising at reset raises before the hold, which resets last, takes a thread state."""
    before = _thread_states()

    with pytest.raises(RuntimeError, match="SWIG director method error"):
        Leo500(ImuSpacecraft()).run(_Remembering(), {}, duration_s=3.0, ground=_RaisingAtReset())

    assert _thread_states() == before, "a run that raised at reset left a thread state behind"


def test_the_thread_state_is_held_at_the_lowest_priority_of_the_run():
    """The hold's process ranks below every other process of a run, the ground's included, and not below -1."""
    scenario = Leo500(ImuSpacecraft())
    simulation = scenario._simulation

    scenario.run(_Remembering(), {}, duration_s=1.0, ground=_Remembering())

    priorities = {process.Name: process.processData.processPriority for process in simulation.procList}
    hold = priorities.pop("thread_state_process")
    assert hold < min(priorities.values()), f"the hold at {hold} is not below {priorities}"
    assert hold >= -1, f"the hold at {hold} is below -1, which Basilisk does not reach at a stop time"


def test_a_run_of_no_duration_holds_no_thread_state():
    """A run of 0 s, whose stop time no task period could reach, has no thread-state process."""
    scenario = Leo500(ImuSpacecraft())
    simulation = scenario._simulation

    scenario.run(_Remembering(), {}, duration_s=0.0)

    names = [process.Name for process in simulation.procList]
    assert "thread_state_process" not in names, f"a run of no duration made a thread-state process: {names}"


def test_the_thread_state_is_given_back_only_on_the_thread_that_took_it():
    """Giving the held thread state back on another thread than the one that took it raises."""
    hold = ThreadStateHold()
    taken = threading.Thread(target=hold.Reset, args=(0,))
    taken.start()
    taken.join()

    with pytest.raises(RuntimeError, match="another thread"):
        hold.UpdateState(0)


def test_the_thread_state_is_given_back_once_at_the_stop_time(monkeypatch: pytest.MonkeyPatch):
    """The hold's task fires once, at the run's final instant.

    Args:
        monkeypatch: Swaps a hold that notes its calls in for the run's.
    """
    calls: list[int] = []

    class _Counting(ThreadStateHold):
        """The hold, noting the time of each call to its `UpdateState`."""

        def UpdateState(self, CurrentSimNanos: int) -> None:
            """Note the call's time, then give the thread state back.

            Args:
                CurrentSimNanos: The simulation time, in nanoseconds.
            """
            calls.append(CurrentSimNanos)
            super().UpdateState(CurrentSimNanos)

    monkeypatch.setattr(scenario_module, "ThreadStateHold", _Counting)

    Leo500(ImuSpacecraft()).run(_Remembering(), {}, duration_s=3.0)

    assert calls == [3_000_000_000], f"the hold was called at {calls}"
