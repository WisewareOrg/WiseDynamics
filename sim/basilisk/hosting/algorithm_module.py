"""A generic Basilisk Python `SysModel` that hosts one framework-free algorithm. The caller
supplies every message, field, frame transform, and algorithm-construction detail through
`FieldMap` lists and two callables.
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from Basilisk.architecture import sysModel
from Basilisk.utilities import macros

# The input-side frame transform the host performs: rotating a reading in its own mounting
# frame into the frame the algorithm's sample expects, via that input's registered mounting.
_SUPPORTED_FRAMES = (None, "P->B")


@dataclass(frozen=True)
class FieldMap:
    """One field of a message payload mapped to/from one field of the algorithm's sample or
    state. For `input_map`, `src` is `"<input name>.<PayloadField>"` and `dst` is a field name
    of `sample_type`. For `output_map`, `src` is a field name of the algorithm's state object
    and `dst` is `"<output name>.<PayloadField>"`. `frame="P->B"` multiplies the value by the
    transpose of that map's input's registered mounting (see `mountings`) before it reaches
    `dst`; only an input map carries a `frame`."""

    src: str
    dst: str
    frame: str | None = None


class AlgorithmModule(sysModel.SysModel):
    """Hosts one framework-free algorithm instance inside a Basilisk task. `Reset` builds the
    algorithm via `make_algorithm(t0, init_from())` and writes its initial state to every
    output; `UpdateState` steps once `t` exceeds the algorithm's current state time (the call
    Basilisk makes at the same sim time as `Reset` carries `dt = 0`), reading every input once,
    mapping fields into one `sample_type` instance per `input_map`, calling
    `algorithm.step(sample)`, and writing the result to every output per `output_map`. The
    built algorithm instance is exposed as `.algorithm` for recorders and dispersions to reach
    inside it."""

    def __init__(
        self,
        inputs: dict[str, Any],
        outputs: dict[str, tuple[type, Any]],
        input_map: list[FieldMap],
        output_map: list[FieldMap],
        sample_type: type,
        make_algorithm: Callable[[float, dict], Any],
        init_from: Callable[[], dict],
        mountings: dict[str, np.ndarray] | None = None,
    ) -> None:
        super().__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.input_map = input_map
        self.output_map = output_map
        self.sample_type = sample_type
        self.make_algorithm = make_algorithm
        self.init_from = init_from
        self.mountings = {
            name: np.array(dcm, dtype=float) for name, dcm in (mountings or {}).items()
        }
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
            if fm.frame == "P->B" and input_name not in self.mountings:
                raise ValueError(
                    f"field map {fm!r} names frame 'P->B' but input {input_name!r} has no "
                    "registered mounting"
                )

        for fm in self.output_map:
            if fm.frame is not None:
                raise ValueError(f"frame transforms apply only to input maps, got {fm!r}")
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
                value = self.mountings[input_name].T @ value
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
