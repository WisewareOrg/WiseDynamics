"""An IMU spacecraft that hands out no truth messages."""

from wise_dynamics.basilisk import ImuSpacecraft
from wise_dynamics.basilisk.bsk_types import Message


class SilentSpacecraft(ImuSpacecraft):
    """`ImuSpacecraft` whose `truth_messages()` is empty, so a case of it alone records nothing."""

    def truth_messages(self) -> dict[str, Message]:
        return {}
