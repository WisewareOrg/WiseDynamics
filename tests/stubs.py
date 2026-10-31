"""A stub Basilisk flight-software module: echoes a fixed state to one output message. Stands
in for a real flight-software module in this repository's own tests."""

from Basilisk.architecture import messaging, sysModel


class StubFsw(sysModel.SysModel):
    """Writes a fixed `sigma_BN` to `output_msg` every step."""

    FIXED_SIGMA_BN = [0.1, 0.2, 0.3]

    def __init__(self) -> None:
        super().__init__()
        self.output_msg = messaging.NavAttMsg()

    def UpdateState(self, CurrentSimNanos: int) -> None:
        payload = messaging.NavAttMsgPayload()
        payload.sigma_BN = self.FIXED_SIGMA_BN
        self.output_msg.write(payload, CurrentSimNanos, self.moduleID)
