"""A stub Basilisk flight-software module: echoes a fixed state to one output message, in the
`{role: (payload_type, msg)}` shape `run_case` expects from any host-adapter-built module.
Stands in for a real flight-software module so this repository's own tests need no algorithm
dependency."""

from Basilisk.architecture import messaging, sysModel


class StubFsw(sysModel.SysModel):
    """Writes a fixed `sigma_BN` to its one output message, `"stub_out"`, every step."""

    FIXED_SIGMA_BN = [0.1, 0.2, 0.3]

    def __init__(self) -> None:
        super().__init__()
        self.outputs = {"stub_out": (messaging.NavAttMsgPayload, messaging.NavAttMsg())}

    def UpdateState(self, CurrentSimNanos: int) -> None:
        payload = messaging.NavAttMsgPayload()
        payload.sigma_BN = self.FIXED_SIGMA_BN
        _, msg = self.outputs["stub_out"]
        msg.write(payload, CurrentSimNanos, self.moduleID)
