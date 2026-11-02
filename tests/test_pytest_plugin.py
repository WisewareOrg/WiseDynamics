"""The pytest plugin, exercised through real pytest sessions in subprocesses, loaded by its
`pytest11` entry point."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

import tests.support
from wise_dynamics import Campaign

CAMPAIGN_TEST = """
import numpy as np

from support import StubFsw
from wise_dynamics.basilisk import run_case


def test_case(sim_campaign, sim_draws, sim_plant, sim_run_writer):
    for name, value in sim_draws.items():
        assert np.array_equal(getattr(sim_plant.config, name), value)
    stub = StubFsw()
    recording = run_case(
        sim_plant,
        stub,
        fsw_outputs={"stub": stub.output_msg},
        duration_s=sim_campaign.duration_s,
        record_s=1.0,
    )
    sim_run_writer(recording.datasets(sim_plant.recorded_fields))
"""

SHORT = ("--sim-duration-s", "3")


@pytest.fixture
def campaign_dir(pytester: pytest.Pytester) -> pytest.Pytester:
    """A pytester directory holding the stub module and a one-test campaign."""
    pytester.makepyfile(support=Path(tests.support.__file__).read_text())
    pytester.makepyfile(test_campaign=CAMPAIGN_TEST)
    return pytester


def run_files(out_dir: Path) -> list[h5py.File]:
    return [h5py.File(path, "r") for path in sorted(out_dir.glob("run_*.h5"))]


def draws_of(run_file: h5py.File) -> dict[str, np.ndarray]:
    return {
        name.removeprefix("draw_"): np.array(value)
        for name, value in run_file.attrs.items()
        if name.startswith("draw_")
    }


def test_each_case_records_the_standard_draws_its_plant_used(campaign_dir):
    out_dir = campaign_dir.path / "out"
    result = campaign_dir.runpytest_subprocess("--sim-runs", "3", "--sim-out", str(out_dir), *SHORT)

    result.assert_outcomes(passed=3)
    campaign = Campaign("imu_spacecraft", seed=0, runs=3, duration_s=3.0, out_dir=out_dir)
    files = run_files(out_dir)
    assert [f.attrs["case"] for f in files] == [0, 1, 2]
    for index, run_file in enumerate(files):
        draws = draws_of(run_file)
        expected = campaign.standard_draws(index)
        assert draws.keys() == expected.keys()
        for name, value in expected.items():
            assert np.array_equal(draws[name], value), (index, name)
        assert np.array_equal(run_file["truth/sigma_BN"][0], draws["sigma0_BN"])
        assert np.allclose(
            run_file["truth/omega_BN_B"][0], draws["omega0_BN_B"], rtol=0, atol=1e-15
        )
    for name in draws_of(files[0]):
        assert not np.allclose(draws_of(files[0])[name], draws_of(files[1])[name])


def test_cases_with_equal_draws_record_distinct_sensor_noise(campaign_dir):
    """With every case at the nominal vehicle, the gyro record differs between cases through
    each case's own noise seed alone."""
    campaign_dir.makeconftest(
        """
        import pytest

        @pytest.fixture
        def sim_draws():
            return {}
        """
    )
    out_dir = campaign_dir.path / "out"
    campaign_dir.runpytest_subprocess(
        "--sim-runs", "2", "--sim-out", str(out_dir), *SHORT
    ).assert_outcomes(passed=2)

    first, second = run_files(out_dir)
    assert not draws_of(first) and not draws_of(second)
    assert np.array_equal(first["truth/sigma_BN"][:], second["truth/sigma_BN"][:])
    assert not np.allclose(first["imu/gyro_P"][1:], second["imu/gyro_P"][1:])


def test_a_scenario_supplying_its_own_draws_records_what_its_plant_used(campaign_dir):
    campaign_dir.makeconftest(
        """
        import pytest

        @pytest.fixture
        def sim_draws(sim_campaign, sim_case_index):
            rng = sim_campaign.draws_rng(sim_case_index)
            return {"gyro_bias_P": rng.normal(0.0, 1e-3, size=3)}
        """
    )
    out_dir = campaign_dir.path / "out"
    campaign_dir.runpytest_subprocess(
        "--sim-runs", "2", "--sim-out", str(out_dir), *SHORT
    ).assert_outcomes(passed=2)

    campaign = Campaign("imu_spacecraft", seed=0, runs=2, duration_s=3.0, out_dir=out_dir)
    for index, run_file in enumerate(run_files(out_dir)):
        expected = campaign.draws_rng(index).normal(0.0, 1e-3, size=3)
        assert draws_of(run_file).keys() == {"gyro_bias_P"}
        assert np.array_equal(draws_of(run_file)["gyro_bias_P"], expected)


def test_a_campaign_reproduces_from_its_seed(campaign_dir):
    def campaign_data(seed: str, name: str) -> list[dict[str, np.ndarray]]:
        out_dir = campaign_dir.path / name
        campaign_dir.runpytest_subprocess(
            "--sim-runs", "2", "--sim-seed", seed, "--sim-out", str(out_dir), *SHORT
        ).assert_outcomes(passed=2)
        return [
            {**draws_of(f), **{key: f[key][:] for key in ("imu/gyro_P", "truth/sigma_BN")}}
            for f in run_files(out_dir)
        ]

    first, again, other = campaign_data("3", "a"), campaign_data("3", "b"), campaign_data("4", "c")
    for case_first, case_again, case_other in zip(first, again, other, strict=True):
        for key, value in case_first.items():
            assert np.array_equal(value, case_again[key]), key
            assert not np.array_equal(value, case_other[key]), key


def test_consumer_fixtures_with_common_names_leave_the_campaign_unchanged(campaign_dir):
    campaign_dir.makeconftest(
        """
        import numpy as np
        import pytest

        @pytest.fixture
        def rng():
            return np.random.default_rng(1234)

        @pytest.fixture
        def seed():
            return 99

        @pytest.fixture
        def out_dir(tmp_path):
            return tmp_path

        @pytest.fixture
        def plant():
            return None

        @pytest.fixture
        def draws():
            return {}
        """
    )
    out_dir = campaign_dir.path / "out"
    campaign_dir.runpytest_subprocess(
        "--sim-runs", "2", "--sim-out", str(out_dir), *SHORT
    ).assert_outcomes(passed=2)

    campaign = Campaign("imu_spacecraft", seed=0, runs=2, duration_s=3.0, out_dir=out_dir)
    for index, run_file in enumerate(run_files(out_dir)):
        for name, value in campaign.standard_draws(index).items():
            assert np.array_equal(draws_of(run_file)[name], value)


def test_campaign_from_config_returns_the_options(pytester, tmp_path):
    report = tmp_path / "campaign.json"
    pytester.makepyfile(
        f"""
        import json

        from wise_dynamics.pytest_plugin import campaign_from_config

        def test_campaign(request, sim_campaign):
            campaign = campaign_from_config(request.config)
            assert campaign is sim_campaign
            fields = ("plant", "seed", "runs", "duration_s", "out_dir")
            values = {{name: str(getattr(campaign, name)) for name in fields}}
            open({str(report)!r}, "w").write(json.dumps(values))
        """
    )
    pytester.runpytest_subprocess(
        "--sim-runs",
        "3",
        "--sim-seed",
        "7",
        "--sim-duration-s",
        "12.5",
        "--sim-spacecraft",
        "imu_spacecraft",
        "--sim-out",
        "relative/out",
    ).assert_outcomes(passed=1)

    assert json.loads(report.read_text()) == {
        "plant": "imu_spacecraft",
        "seed": "7",
        "runs": "3",
        "duration_s": "12.5",
        "out_dir": str(pytester.path / "relative" / "out"),
    }


def test_workers_share_the_default_out_dir(campaign_dir, monkeypatch):
    """`TMPDIR` points the default directory into this test's own directory, where it can be
    found."""
    monkeypatch.setenv("TMPDIR", str(campaign_dir.path))
    result = campaign_dir.runpytest_subprocess("-n", "2", "--sim-runs", "2", *SHORT)

    result.assert_outcomes(passed=2)
    files = list(campaign_dir.path.rglob("run_*.h5"))
    assert len(files) == 2
    assert len({path.parent for path in files}) == 1
    result.stdout.fnmatch_lines([f"wise-dynamics: 2 run files in {files[0].parent}"])


def test_workers_share_a_relative_out_dir_resolved_by_the_controller(campaign_dir):
    """Each worker changes directory before the plugin configures it; the run files still land
    in the directory the controller resolved."""
    campaign_dir.makeconftest(
        """
        import os
        import pytest

        @pytest.hookimpl(tryfirst=True)
        def pytest_configure(config):
            if hasattr(config, "workerinput"):
                worker_dir = config.rootpath / config.workerinput["workerid"]
                worker_dir.mkdir()
                os.chdir(worker_dir)
        """
    )
    campaign_dir.runpytest_subprocess(
        "-n", "2", "--sim-runs", "2", "--sim-out", "out", *SHORT
    ).assert_outcomes(passed=2)
    assert len(list((campaign_dir.path / "out").glob("run_*.h5"))) == 2


def test_a_campaign_runs_without_xdist(campaign_dir):
    out_dir = campaign_dir.path / "out"
    campaign_dir.runpytest_subprocess(
        "-p", "no:xdist", "--sim-runs", "1", "--sim-out", str(out_dir), *SHORT
    ).assert_outcomes(passed=1)
    assert len(list(out_dir.glob("run_*.h5"))) == 1


def test_a_session_with_no_case_creates_no_directory(pytester, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(pytester.path))
    pytester.makepyfile("def test_nothing():\n    pass\n")
    pytester.runpytest_subprocess().assert_outcomes(passed=1)
    assert not list(pytester.path.glob("wise_dynamics_*"))


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--sim-runs", "0", "a campaign needs at least one run, got 0"),
        ("--sim-duration-s", "0", "case duration must be positive, got 0.0 s"),
        ("--sim-seed", "-1", "campaign seed must be non-negative, got -1"),
    ],
)
def test_out_of_range_options_are_usage_errors(pytester, option, value, message):
    pytester.makepyfile("def test_case(sim_case_index):\n    pass\n")
    result = pytester.runpytest_subprocess(option, value)
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"ERROR: wise-dynamics: {message}"])


def test_an_unknown_spacecraft_fails_collection_naming_it_and_the_known_ones(pytester):
    pytester.makepyfile("def test_case(sim_case_index):\n    pass\n")
    result = pytester.runpytest_subprocess("--sim-spacecraft", "bogus")
    assert result.ret == pytest.ExitCode.INTERRUPTED
    result.stdout.fnmatch_lines(
        [
            "*UsageError: wise-dynamics: --sim-spacecraft: unknown plant 'bogus'; "
            "known plants: imu_spacecraft"
        ]
    )


def test_an_out_dir_holding_run_files_is_a_usage_error(pytester):
    out_dir = pytester.path / "out"
    out_dir.mkdir()
    (out_dir / "run_0000.h5").write_bytes(b"")
    pytester.makepyfile("def test_case(sim_case_index):\n    pass\n")
    result = pytester.runpytest_subprocess("--sim-out", str(out_dir))
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines([f"ERROR: --sim-out {out_dir} already holds 1 run files*"])


def test_consumer_options_with_short_names_coexist_with_the_plugin(pytester):
    pytester.makeconftest(
        """
        def pytest_addoption(parser):
            for name in ("--runs", "--out", "--seed", "--duration-s", "--spacecraft"):
                parser.addoption(name, action="store")
        """
    )
    pytester.makepyfile("def test_nothing():\n    pass\n")
    pytester.runpytest_subprocess("--out", "x").assert_outcomes(passed=1)
