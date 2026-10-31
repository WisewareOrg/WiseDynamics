"""Producer-side conformance: the campaign's own export against WiseGncAnalysis's real
`contract.v0.check_campaign`, not a restated assumption about the columns it writes."""

from contract.v0 import check_campaign

from sim.basilisk.campaigns.imu_dead_reckoning_mc import build_controller, export_campaign


def test_eight_run_export_passes_the_real_contract_check(tmp_path):
    controller = build_controller(
        num_runs=8, archive_dir=str(tmp_path / "mc_contract"), worker_count=8
    )
    failures = controller.executeSimulations()
    assert failures == [], f"campaign had failures: {failures}"

    out_dir = tmp_path / "mc_contract_export"
    export_campaign(controller, 8, out_dir)

    problems = check_campaign(out_dir)
    assert problems == [], f"contract violations: {problems}"
