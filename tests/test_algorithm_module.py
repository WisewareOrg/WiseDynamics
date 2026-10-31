"""Unit tests for `AlgorithmModule` itself: no dead-reckoning algorithm, no full sim. A stub
algorithm and standalone Basilisk messages (written directly, not through a running task) are
enough to drive `Reset`/`UpdateState` and inspect what comes out."""

from dataclasses import dataclass, field

import numpy as np
import pytest
from Basilisk.architecture import messaging

from sim.basilisk.hosting.algorithm_module import AlgorithmModule, FieldMap


@dataclass
class StubSample:
    t: float
    a: np.ndarray
    b: np.ndarray


@dataclass
class StubState:
    t: float
    value: np.ndarray = field(default_factory=lambda: np.zeros(3))


class StubAlgorithm:
    """Records every sample it receives; always steps to a fixed, recognizable state so a test
    can tell a step happened without caring about any real algorithm's numerics."""

    FIXED_VALUE = np.array([1.0, 2.0, 3.0])

    def __init__(self, t0: float, init: dict) -> None:
        self.samples: list[StubSample] = []
        self._state = StubState(t=t0, value=np.array(init.get("value0", [0.0, 0.0, 0.0])))

    @property
    def state(self) -> StubState:
        return self._state

    def step(self, sample: StubSample) -> StubState:
        self.samples.append(sample)
        self._state = StubState(t=sample.t, value=self.FIXED_VALUE)
        return self._state


def _two_input_two_output_module(mountings=None) -> AlgorithmModule:
    return AlgorithmModule(
        inputs={"imu": messaging.IMUSensorMsgReader(), "nav": messaging.NavAttMsgReader()},
        outputs={
            "outAtt": (messaging.NavAttMsgPayload, messaging.NavAttMsg()),
            "outTrans": (messaging.NavTransMsgPayload, messaging.NavTransMsg()),
        },
        input_map=[
            FieldMap("imu.AngVelPlatform", "a", frame="P->B"),
            FieldMap("nav.sigma_BN", "b"),
        ],
        output_map=[
            FieldMap("value", "outAtt.sigma_BN"),
            FieldMap("value", "outTrans.r_BN_N"),
        ],
        sample_type=StubSample,
        make_algorithm=lambda t0, init: StubAlgorithm(t0, init),
        init_from=lambda: {"value0": [0.0, 0.0, 0.0]},
        mountings=mountings if mountings is not None else {"imu": np.eye(3)},
    )


def _write_imu(angvel) -> messaging.IMUSensorMsg:
    msg = messaging.IMUSensorMsg()
    payload = messaging.IMUSensorMsgPayload()
    payload.AngVelPlatform = list(angvel)
    msg.write(payload, 0, 0)
    return msg


def _write_nav(sigma) -> messaging.NavAttMsg:
    msg = messaging.NavAttMsg()
    payload = messaging.NavAttMsgPayload()
    payload.sigma_BN = list(sigma)
    msg.write(payload, 0, 0)
    return msg


def test_field_mapping_lands_the_right_numbers_in_the_right_sample_fields():
    module = _two_input_two_output_module()
    module.inputs["imu"].subscribeTo(_write_imu([1.0, 2.0, 3.0]))
    module.inputs["nav"].subscribeTo(_write_nav([0.4, 0.5, 0.6]))

    module.Reset(0)
    module.UpdateState(1_000_000_000)  # 1 s later

    sample = module.algorithm.samples[-1]
    assert np.allclose(sample.a, [1.0, 2.0, 3.0])  # identity mounting: unchanged
    assert np.allclose(sample.b, [0.4, 0.5, 0.6])  # unmapped frame: passed through raw


def test_p_to_b_frame_applies_the_transpose_of_a_non_identity_mounting():
    # 90 deg rotation about +Z: platform-frame x becomes body-frame +y under this matrix's
    # transpose.
    dcm_PB = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    module = _two_input_two_output_module(mountings={"imu": dcm_PB})
    module.inputs["imu"].subscribeTo(_write_imu([1.0, 0.0, 0.0]))
    module.inputs["nav"].subscribeTo(_write_nav([0.0, 0.0, 0.0]))

    module.Reset(0)
    module.UpdateState(1_000_000_000)

    expected = dcm_PB.T @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(module.algorithm.samples[-1].a, expected)
    assert not np.allclose(expected, [1.0, 0.0, 0.0])  # the transform actually did something


def test_two_inputs_and_two_outputs_work_together():
    module = _two_input_two_output_module()
    out_att_msg = module.outputs["outAtt"][1]
    out_trans_msg = module.outputs["outTrans"][1]
    module.inputs["imu"].subscribeTo(_write_imu([1.0, 2.0, 3.0]))
    module.inputs["nav"].subscribeTo(_write_nav([0.4, 0.5, 0.6]))

    module.Reset(0)
    module.UpdateState(1_000_000_000)

    assert np.allclose(out_att_msg.read().sigma_BN, StubAlgorithm.FIXED_VALUE)
    assert np.allclose(out_trans_msg.read().r_BN_N, StubAlgorithm.FIXED_VALUE)


def test_construction_raises_on_a_misspelled_field():
    with pytest.raises(ValueError, match="AngVelPltaform"):
        AlgorithmModule(
            inputs={"imu": messaging.IMUSensorMsgReader()},
            outputs={"outAtt": (messaging.NavAttMsgPayload, messaging.NavAttMsg())},
            input_map=[FieldMap("imu.AngVelPltaform", "a")],  # misspelled
            output_map=[FieldMap("value", "outAtt.sigma_BN")],
            sample_type=StubSample,
            make_algorithm=lambda t0, init: StubAlgorithm(t0, init),
            init_from=lambda: {},
        )


def test_construction_raises_on_a_frame_in_an_output_map():
    with pytest.raises(ValueError, match="apply only to input maps"):
        AlgorithmModule(
            inputs={"imu": messaging.IMUSensorMsgReader()},
            outputs={"outAtt": (messaging.NavAttMsgPayload, messaging.NavAttMsg())},
            input_map=[FieldMap("imu.AngVelPlatform", "a")],
            output_map=[FieldMap("value", "outAtt.sigma_BN", frame="P->B")],
            sample_type=StubSample,
            make_algorithm=lambda t0, init: StubAlgorithm(t0, init),
            init_from=lambda: {},
        )


def test_construction_raises_on_p_to_b_for_an_input_with_no_registered_mounting():
    with pytest.raises(ValueError, match="no registered mounting"):
        AlgorithmModule(
            inputs={"imu": messaging.IMUSensorMsgReader()},
            outputs={"outAtt": (messaging.NavAttMsgPayload, messaging.NavAttMsg())},
            input_map=[FieldMap("imu.AngVelPlatform", "a", frame="P->B")],
            output_map=[FieldMap("value", "outAtt.sigma_BN")],
            sample_type=StubSample,
            make_algorithm=lambda t0, init: StubAlgorithm(t0, init),
            init_from=lambda: {},
            mountings={},
        )


def test_zero_dt_first_call_does_not_step():
    module = _two_input_two_output_module()
    module.inputs["imu"].subscribeTo(_write_imu([1.0, 2.0, 3.0]))
    module.inputs["nav"].subscribeTo(_write_nav([0.4, 0.5, 0.6]))

    module.Reset(0)
    assert len(module.algorithm.samples) == 0

    module.UpdateState(0)  # same sim time as Reset: dt would be zero
    assert len(module.algorithm.samples) == 0

    module.UpdateState(1_000_000_000)  # real dt: now it steps
    assert len(module.algorithm.samples) == 1
