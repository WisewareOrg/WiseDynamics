"""A stub flight-software module and a helper that runs a plant with it."""

from Basilisk.architecture import messaging, sysModel

from wise_dynamics.basilisk import Plant, Recording, run_case


class StubFsw(sysModel.SysModel):
    """Writes `FIXED_SIGMA_BN` to `output_msg`, a `NavAttMsg`, every step."""

    FIXED_SIGMA_BN = [0.1, 0.2, 0.3]

    def __init__(self) -> None:
        super().__init__()
        self.ModelTag = "stub"
        self.output_msg = messaging.NavAttMsg()

    def UpdateState(self, CurrentSimNanos: int) -> None:
        payload = messaging.NavAttMsgPayload()
        payload.sigma_BN = self.FIXED_SIGMA_BN
        self.output_msg.write(payload, CurrentSimNanos, self.moduleID)


def run_with_stub(plant: Plant, duration_s: float, record_s: float = 10.0) -> Recording:
    """Run `plant` with a `StubFsw`, recording its output under role `stub`."""
    stub = StubFsw()
    return run_case(
        plant,
        stub,
        fsw_outputs={"stub": stub.output_msg},
        duration_s=duration_s,
        record_s=record_s,
    )
