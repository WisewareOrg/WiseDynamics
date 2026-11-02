"""The pytest plugin, exercised through real pytest sessions in subprocesses, loaded by its
`pytest11` entry point."""

import re
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pytest

import tests.support
from wise_dynamics.basilisk import ImuSpacecraft

CAMPAIGN_TEST = """
import numpy as np

from support import StubFsw
from wise_dynamics.basilisk import run_case


def test_case(sim_duration_s, sim_plant, sim_run_writer):
    stub = StubFsw()
    recording = run_case(
        sim_plant,
        stub,
        fsw_outputs={"stub": stub.output_msg},
        duration_s=sim_duration_s,
        record_s=1.0,
    )
    sim_run_writer(
        recording.datasets(sim_plant.recorded_fields),
        extra_attrs={
            "probe_noise_seed": sim_plant.imu.RNGSeed,
            "probe_gyro_bias_P": np.array(sim_plant.imu.senRotBias, dtype=float).flatten(),
        },
    )
"""

SHORT = ("--sim-duration-s", "3")

# Case 5 of seed 0, as wise-dynamics recorded it at 04150b2: every campaign recorded since
# reproduces only while these hold.
CASE_5_NOISE_SEED = 2153058428
CASE_5_DRAWS = {
    "sigma0_BN": [0.10274604175442446, 0.19692436106593408, -0.3223839260439085],
    "omega0_BN_B": [0.012184953200484915, 0.01815985986877613, 0.014939997512415095],
    "gyro_bias_P": [8.079773541256839e-05, -0.00011052054402547108, -5.069458483787486e-05],
    "accel_bias_P": [0.00012226550330897274, 1.6712413497495933e-05, 9.917025892881453e-05],
}


@pytest.fixture
def campaign_dir(pytester: pytest.Pytester) -> pytest.Pytester:
    """A pytester directory holding the stub module and a one-test campaign."""
    pytester.makepyfile(support=Path(tests.support.__file__).read_text())
    pytester.makepyfile(test_campaign=CAMPAIGN_TEST)
    return pytester


def run_campaign(campaign_dir: pytest.Pytester, out: str, *args: str, passed: int) -> list[dict]:
    """Run the campaign into `out` and return each run file's attributes and datasets."""
    out_dir = campaign_dir.path / out
    campaign_dir.runpytest_subprocess("--sim-out", str(out_dir), *SHORT, *args).assert_outcomes(
        passed=passed
    )
    runs = []
    for path in sorted(out_dir.glob("run_*.h5")):
        with h5py.File(path, "r") as run_file:
            runs.append(
                {
                    "name": path.name,
                    "attrs": {key: run_file.attrs[key] for key in run_file.attrs},
                    "t_s": run_file["t_s"][:],
                    "gyro_P": run_file["imu/gyro_P"][:],
                    "sigma_BN": run_file["truth/sigma_BN"][:],
                    "omega_BN_B": run_file["truth/omega_BN_B"][:],
                }
            )
    return runs


def draws_of(run: dict) -> dict[str, np.ndarray]:
    return {
        key.removeprefix("draw_"): np.asarray(value)
        for key, value in run["attrs"].items()
        if key.startswith("draw_")
    }


def standard_draws(seed: int, index: int) -> dict[str, np.ndarray]:
    """The standard draws of case `index`, derived as the README states."""
    draws_stream, _ = np.random.SeedSequence(seed, spawn_key=(index,)).spawn(2)
    rng = np.random.default_rng(draws_stream)
    return ImuSpacecraft.standard_dispersion.draw(ImuSpacecraft.nominal(), rng)


def test_case_seeding_matches_recorded_campaigns(campaign_dir):
    (run,) = run_campaign(
        campaign_dir, "out", "--sim-runs", "852", "test_campaign.py::test_case[5]", passed=1
    )
    assert run["name"] == "run_0005.h5"
    assert run["attrs"]["probe_noise_seed"] == CASE_5_NOISE_SEED
    assert {name: value.tolist() for name, value in draws_of(run).items()} == CASE_5_DRAWS


def test_a_case_is_the_same_whatever_the_run_count(campaign_dir):
    (small,) = run_campaign(
        campaign_dir, "small", "--sim-runs", "8", "test_campaign.py::test_case[5]", passed=1
    )
    (large,) = run_campaign(
        campaign_dir, "large", "--sim-runs", "852", "test_campaign.py::test_case[5]", passed=1
    )
    assert small["attrs"]["probe_noise_seed"] == large["attrs"]["probe_noise_seed"]
    for name, value in draws_of(small).items():
        assert np.array_equal(value, draws_of(large)[name])


def test_noise_seeds_differ_across_cases_and_campaign_seeds(campaign_dir):
    seeds = [
        run["attrs"]["probe_noise_seed"]
        for seed in ("0", "1")
        for run in run_campaign(
            campaign_dir, f"seed{seed}", "--sim-runs", "2", "--sim-seed", seed, passed=2
        )
    ]
    assert len(set(seeds)) == 4, seeds


def test_each_case_records_the_standard_draws_its_plant_used(campaign_dir):
    runs = run_campaign(campaign_dir, "out", "--sim-runs", "3", passed=3)

    assert [run["attrs"]["case"] for run in runs] == [0, 1, 2]
    for index, run in enumerate(runs):
        draws = draws_of(run)
        expected = standard_draws(0, index)
        assert draws.keys() == expected.keys()
        for name, value in expected.items():
            assert np.array_equal(draws[name], value), (index, name)
        assert np.array_equal(run["attrs"]["probe_gyro_bias_P"], draws["gyro_bias_P"])
        assert np.array_equal(run["sigma_BN"][0], draws["sigma0_BN"])
        assert np.allclose(run["omega_BN_B"][0], draws["omega0_BN_B"], rtol=0, atol=1e-15)
    for name in draws_of(runs[0]):
        assert not np.allclose(draws_of(runs[0])[name], draws_of(runs[1])[name])


def test_the_run_file_carries_the_case_root_attrs(campaign_dir):
    runs = run_campaign(campaign_dir, "out", "--sim-runs", "2", "--sim-seed", "42", passed=2)

    run = runs[1]
    attrs = run["attrs"]
    assert run["name"] == "run_0001.h5"
    assert attrs["case"] == 1
    assert attrs["seed"] == 42
    assert attrs["producer"] == "wise_dynamics.basilisk"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", attrs["created_utc"])
    assert attrs["versions_bsk"] == version("bsk")
    assert attrs["versions_wise_dynamics"].startswith(version("wise-dynamics"))
    assert "probe_noise_seed" in attrs


def test_extra_attrs_cannot_replace_a_root_attr(pytester):
    pytester.makepyfile(
        """
        import numpy as np
        import pytest

        from wise_dynamics import Dataset

        def test_case(sim_run_writer):
            with pytest.raises(ValueError, match=r"reserved root attrs \\['seed'\\]"):
                sim_run_writer({"t_s": Dataset(np.zeros(1), "s")}, extra_attrs={"seed": 1})
        """
    )
    pytester.runpytest_subprocess(
        "--sim-runs", "1", "--sim-out", str(pytester.path / "out")
    ).assert_outcomes(passed=1)


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
    first, second = run_campaign(campaign_dir, "out", "--sim-runs", "2", passed=2)

    assert not draws_of(first) and not draws_of(second)
    assert np.array_equal(first["sigma_BN"], second["sigma_BN"])
    assert not np.allclose(first["gyro_P"][1:], second["gyro_P"][1:])


OWN_DRAWS_CONFTEST = """
import pytest

@pytest.fixture
def sim_draws(sim_draws_rng):
    return {"gyro_bias_P": sim_draws_rng.normal(0.0, 1e-3, size=3)}
"""


def case_streams(seed: int, index: int) -> list[np.random.SeedSequence]:
    """Case `index`'s two streams, derived as the README states."""
    return np.random.SeedSequence(seed, spawn_key=(index,)).spawn(2)


def test_a_scenario_supplying_its_own_draws_records_what_its_plant_used(campaign_dir):
    """The plant is built from the scenario's draws with the case's own noise seed, and the
    run file records exactly those draws."""
    campaign_dir.makeconftest(OWN_DRAWS_CONFTEST)
    runs = run_campaign(campaign_dir, "out", "--sim-runs", "2", passed=2)

    for index, run in enumerate(runs):
        draws_stream, noise_stream = case_streams(0, index)
        expected = np.random.default_rng(draws_stream).normal(0.0, 1e-3, size=3)
        assert draws_of(run).keys() == {"gyro_bias_P"}
        assert np.array_equal(draws_of(run)["gyro_bias_P"], expected)
        assert np.array_equal(run["attrs"]["probe_gyro_bias_P"], expected)
        assert run["attrs"]["probe_noise_seed"] == int(noise_stream.generate_state(1)[0])


def test_scenario_draws_differ_between_neighbouring_cases_of_neighbouring_seeds(campaign_dir):
    """Case 1 of seed 0 and case 0 of seed 1, the pair a seed-plus-index scheme collides."""
    campaign_dir.makeconftest(OWN_DRAWS_CONFTEST)
    _, case_1_of_seed_0 = run_campaign(campaign_dir, "seed0", "--sim-runs", "2", passed=2)
    case_0_of_seed_1, _ = run_campaign(
        campaign_dir, "seed1", "--sim-runs", "2", "--sim-seed", "1", passed=2
    )
    assert not np.allclose(
        draws_of(case_1_of_seed_0)["gyro_bias_P"], draws_of(case_0_of_seed_1)["gyro_bias_P"]
    )


def test_a_campaign_reproduces_from_its_seed(campaign_dir):
    def campaign(seed: str, out: str) -> list[dict]:
        return run_campaign(campaign_dir, out, "--sim-runs", "2", "--sim-seed", seed, passed=2)

    first, again, other = campaign("3", "a"), campaign("3", "b"), campaign("4", "c")
    for case_first, case_again, case_other in zip(first, again, other, strict=True):
        assert case_first["attrs"]["probe_noise_seed"] == case_again["attrs"]["probe_noise_seed"]
        assert case_first["attrs"]["probe_noise_seed"] != case_other["attrs"]["probe_noise_seed"]
        for name in ("gyro_P", "sigma_BN"):
            assert np.array_equal(case_first[name], case_again[name]), name
            assert not np.array_equal(case_first[name], case_other[name]), name
        for name, value in draws_of(case_first).items():
            assert np.array_equal(value, draws_of(case_again)[name]), name
            assert not np.array_equal(value, draws_of(case_other)[name]), name


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
    runs = run_campaign(campaign_dir, "out", "--sim-runs", "2", passed=2)

    for index, run in enumerate(runs):
        for name, value in standard_draws(0, index).items():
            assert np.array_equal(draws_of(run)[name], value)


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


def test_workers_write_to_the_relative_out_dir_the_controller_resolved(campaign_dir):
    """Each worker changes directory before the plugin configures it, and `--sim-out` is
    relative; the run files still land in the one directory the controller resolved."""
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

    in_out_dir = sorted((campaign_dir.path / "out").glob("run_*.h5"))
    assert len(in_out_dir) == 2
    assert sorted(campaign_dir.path.rglob("run_*.h5")) == in_out_dir


def test_each_case_runs_for_sim_duration_s(campaign_dir):
    (run,) = run_campaign(campaign_dir, "out", "--sim-runs", "1", "--sim-duration-s", "4", passed=1)
    assert run["t_s"][-1] == 4.0


def test_a_campaign_runs_without_xdist(campaign_dir):
    runs = run_campaign(campaign_dir, "out", "-p", "no:xdist", "--sim-runs", "1", passed=1)
    assert len(runs) == 1


def test_a_session_with_no_case_creates_nothing_in_the_temporary_directory(
    pytester, tmp_path, monkeypatch
):
    """The default output directory sits under `TMPDIR`; a session with no case leaves `TMPDIR`
    as it found it."""
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    pytester.makepyfile("def test_nothing():\n    pass\n")
    before = sorted(temporary.iterdir())
    pytester.runpytest_subprocess(f"--basetemp={tmp_path / 'basetemp'}").assert_outcomes(passed=1)
    assert sorted(temporary.iterdir()) == before


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("--sim-runs", "0", "--sim-runs must be at least 1, got 0"),
        ("--sim-duration-s", "0", "--sim-duration-s must be positive, got 0.0"),
        ("--sim-seed", "-1", "--sim-seed must be non-negative, got -1"),
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
