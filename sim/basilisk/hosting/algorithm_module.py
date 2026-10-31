"""A generic Basilisk Python `SysModel` that hosts a framework-free algorithm from
WiseGncDesign. Knows nothing about any particular algorithm: no IMU, gyro, MRP, or dead
reckoning. All algorithm-specific wiring (which messages, which fields, which frame
transforms, how to construct the algorithm) is supplied by the scenario through `FieldMap`
lists and two callables. See `sim/basilisk/scenarios/imu_dead_reckoning.py` for a worked
example of wiring this module to `gnc_algorithms.imu_dead_reckoning.ImuDeadReckoner`.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from Basilisk.architecture import sysModel
from Basilisk.utilities import macros

# The only input-side frame transform the host performs: rotate a sensor's platform-frame
# reading into the body frame via the sensor's own mounting DCM. Anything beyond this single
# transform is an algorithm concern, not the host's.
_SUPPORTED_FRAMES = (None, "P->B")


@dataclass(frozen=True)
class FieldMap:
    """One field of a message payload mapped to/from one field of the algorithm's sample or
    state. For `input_map`, `src` is `"<input name>.<PayloadField>"` and `dst` is a field name
    of `sample_type`. For `output_map`, `src` is a field name of the algorithm's state object
    and `dst` is `"<output name>.<PayloadField>"`. `frame="P->B"` multiplies the value by the
    transpose of the module's `dcm_PB` before it reaches `dst` (input maps only)."""

    src: str
    dst: str
    frame: str | None = None


class AlgorithmModule(sysModel.SysModel):
    """Hosts one framework-free algorithm instance inside a Basilisk task. `Reset` builds the
    algorithm via `make_algorithm(t0, init_from())` and writes its initial state to every
    output; `UpdateState` reads every input once, maps fields into one `sample_type` instance
    per `input_map`, calls `algorithm.step(sample)`, and writes the result to every output per
    `output_map`. Skips stepping on the call Basilisk makes at the same sim time as `Reset`
    (dt would be zero). The built algorithm instance is exposed as `.algorithm` for recorders
    and dispersions that need to reach inside it."""

    def __init__(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, tuple[type, Any]],
        input_map: list[FieldMap],
        output_map: list[FieldMap],
        sample_type: type,
        make_algorithm: Callable[[float, dict], Any],
        init_from: Callable[[], dict],
        dcm_PB: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.input_map = input_map
        self.output_map = output_map
        self.sample_type = sample_type
        self.make_algorithm = make_algorithm
        self.init_from = init_from
        self.dcm_PB = np.eye(3) if dcm_PB is None else np.array(dcm_PB, dtype=float)
        self.algorithm: Any = None

        self._validate_maps()

    def _validate_maps(self) -> None:
        sample_fields = {f.name for f in dataclasses.fields(self.sample_type)}

        for fm in self.input_map:
            if fm.frame not in _SUPPORTED_FRAMES:
                raise ValueError(f"unsupported frame transform {fm.frame!r} in {fm!r}")
            input_name, _, field_name = fm.src.partition(".")
            if not field_name:
                raise ValueError(f"input_map src must be '<input>.<PayloadField>', got {fm.src!r}")
            if input_name not in self.inputs:
                raise ValueError(f"unknown input {input_name!r} in field map src {fm.src!r}")
            zero_payload = self.inputs[input_name].zeroMsgPayload
            if not hasattr(zero_payload, field_name):
                raise ValueError(f"unknown payload field {field_name!r} on input {input_name!r}")
            if fm.dst not in sample_fields:
                raise ValueError(
                    f"unknown field {fm.dst!r} on sample type {self.sample_type.__name__}"
                )

        for fm in self.output_map:
            output_name, _, field_name = fm.dst.partition(".")
            if not field_name:
                raise ValueError(
                    f"output_map dst must be '<output>.<PayloadField>', got {fm.dst!r}"
                )
            if output_name not in self.outputs:
                raise ValueError(f"unknown output {output_name!r} in field map dst {fm.dst!r}")
            payload_type, _ = self.outputs[output_name]
            if not hasattr(payload_type(), field_name):
                raise ValueError(f"unknown payload field {field_name!r} on output {output_name!r}")

    def Reset(self, CurrentSimNanos: int) -> None:
        t0 = CurrentSimNanos * macros.NANO2SEC
        self.algorithm = self.make_algorithm(t0, self.init_from())
        self._write_outputs(self.algorithm.state, CurrentSimNanos)

    def UpdateState(self, CurrentSimNanos: int) -> None:
        t = CurrentSimNanos * macros.NANO2SEC
        if not t > self.algorithm.state.t:
            # Basilisk calls UpdateState once at the same sim time as Reset (dt == 0); the
            # algorithm already holds the correct state for that instant.
            return

        bufs = {name: reader() for name, reader in self.inputs.items()}
        sample_fields: dict[str, Any] = {}
        for fm in self.input_map:
            input_name, _, field_name = fm.src.partition(".")
            value = np.array(getattr(bufs[input_name], field_name), dtype=float)
            if fm.frame == "P->B":
                value = self.dcm_PB.T @ value
            sample_fields[fm.dst] = value

        sample = self.sample_type(t=t, **sample_fields)
        state = self.algorithm.step(sample)
        self._write_outputs(state, CurrentSimNanos)

    def _write_outputs(self, state: Any, CurrentSimNanos: int) -> None:
        payloads: dict[str, Any] = {}
        for fm in self.output_map:
            output_name, _, field_name = fm.dst.partition(".")
            if output_name not in payloads:
                payload_type, _ = self.outputs[output_name]
                payloads[output_name] = payload_type()
            value = getattr(state, fm.src)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            setattr(payloads[output_name], field_name, value)

        for output_name, payload in payloads.items():
            _, msg = self.outputs[output_name]
            msg.write(payload, CurrentSimNanos, self.moduleID)


def hub_initial_state(scObject: Any) -> dict:
    """Reads a spacecraft's live hub initial-condition attributes (which a Monte Carlo run may
    have dispersed before `Reset()` runs). A ready-made `init_from` for the common case of an
    algorithm that starts aligned with the truth hub state; a scenario whose algorithm needs a
    different `init` dict shape passes its own callable instead."""
    return {
        "sigma0_BN": np.array(scObject.hub.sigma_BNInit, dtype=float).flatten(),
        "r0_N": np.array(scObject.hub.r_CN_NInit, dtype=float).flatten(),
        "v0_N": np.array(scObject.hub.v_CN_NInit, dtype=float).flatten(),
        "omega0_BN_B": np.array(scObject.hub.omega_BN_BInit, dtype=float).flatten(),
    }
