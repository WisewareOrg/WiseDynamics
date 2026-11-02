"""`Campaign` and `Case`: seeding, draws, the plant a case builds, and its run file."""

import re
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
import pytest

from wise_dynamics import Campaign, Dataset

# Case 5 of campaign seed 0, as recorded by wise-dynamics at 04150b2: every campaign recorded
# since reproduces only while these hold.
CASE_5_NOISE_SEED = 2153058428
CASE_5_DRAWS = {
    "sigma0_BN": [0.10274604175442446, 0.19692436106593408, -0.3223839260439085],
    "omega0_BN_B": [0.012184953200484915, 0.01815985986877613, 0.014939997512415095],
    "gyro_bias_P": [8.079773541256839e-05, -0.00011052054402547108, -5.069458483787486e-05],
    "accel_bias_P": [0.00012226550330897274, 1.6712413497495933e-05, 9.917025892881453e-05],
}


def make_campaign(tmp_path: Path, *, seed: int = 0, runs: int = 852) -> Campaign:
    return Campaign("imu_spacecraft", seed=seed, runs=runs, duration_s=60.0, out_dir=tmp_path)


def test_case_seeding_matches_recorded_campaigns(tmp_path):
    case = make_campaign(tmp_path).case(5)
    assert case.noise_seed == CASE_5_NOISE_SEED
    assert {name: value.tolist() for name, value in case.draws.items()} == CASE_5_DRAWS


def test_a_case_is_the_same_whatever_the_campaign_size(tmp_path):
    small, large = make_campaign(tmp_path, runs=8).case(5), make_campaign(tmp_path).case(5)
    assert small.noise_seed == large.noise_seed
    for name in small.draws:
        assert np.array_equal(small.draws[name], large.draws[name])


def test_cases_and_seeds_differ(tmp_path):
    first, second = make_campaign(tmp_path).case(0), make_campaign(tmp_path).case(1)
    other_seed = make_campaign(tmp_path, seed=1).case(0)
    assert len({first.noise_seed, second.noise_seed, other_seed.noise_seed}) == 3
    for name in first.draws:
        assert not np.allclose(first.draws[name], second.draws[name])
        assert not np.allclose(first.draws[name], other_seed.draws[name])


def test_a_case_builds_its_plant_from_its_draws_and_noise_seed(tmp_path):
    case = make_campaign(tmp_path).case(5)
    plant = case.build_plant()
    assert plant.config == case.config
    assert np.array_equal(plant.initial_state()["sigma0_BN"], CASE_5_DRAWS["sigma0_BN"])
    assert np.array_equal(np.array(plant.imu.senRotBias).flatten(), CASE_5_DRAWS["gyro_bias_P"])
    assert plant.imu.RNGSeed == CASE_5_NOISE_SEED


def test_given_draws_are_what_the_plant_uses_and_the_run_file_records(tmp_path):
    draws = {"gyro_bias_P": np.array([1e-3, 2e-3, 3e-3])}
    case = make_campaign(tmp_path).case(2, draws)
    assert np.array_equal(np.array(case.build_plant().imu.senRotBias).flatten(), [1e-3, 2e-3, 3e-3])
    with h5py.File(case.write({"t_s": Dataset(np.zeros(1), "s")}), "r") as run_file:
        assert [name for name in run_file.attrs if name.startswith("draw_")] == ["draw_gyro_bias_P"]
        assert np.array_equal(run_file.attrs["draw_gyro_bias_P"], [1e-3, 2e-3, 3e-3])


def test_the_run_file_carries_the_case_root_attrs(tmp_path):
    case = make_campaign(tmp_path, seed=42).case(1)
    path = case.write(
        {"t_s": Dataset(np.zeros(1), "s")}, extra_attrs={"versions_flight_software": "9.9"}
    )

    assert path == tmp_path / "run_0001.h5"
    with h5py.File(path, "r") as run_file:
        attrs = run_file.attrs
        assert attrs["case"] == 1
        assert attrs["seed"] == 42
        assert attrs["producer"] == "wise_dynamics.basilisk"
        assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", attrs["created_utc"])
        assert attrs["versions_bsk"] == version("bsk")
        assert attrs["versions_wise_dynamics"].startswith(version("wise-dynamics"))
        assert attrs["versions_flight_software"] == "9.9"
        for name, value in case.draws.items():
            assert np.array_equal(attrs[f"draw_{name}"], value)


def test_extra_attrs_cannot_replace_a_root_attr(tmp_path):
    case = make_campaign(tmp_path).case(0)
    with pytest.raises(ValueError, match=r"\['seed'\]"):
        case.write({"t_s": Dataset(np.zeros(1), "s")}, extra_attrs={"seed": 1})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"seed": -1}, "seed must be non-negative"),
        ({"runs": 0}, "at least one run"),
        ({"duration_s": 0.0}, "duration must be positive"),
    ],
)
def test_out_of_range_campaigns_are_refused(tmp_path, overrides, message):
    arguments = {"plant": "imu_spacecraft", "seed": 0, "runs": 1, "duration_s": 1.0}
    with pytest.raises(ValueError, match=message):
        Campaign(**{**arguments, **overrides}, out_dir=tmp_path)


def test_a_case_outside_the_campaign_is_refused(tmp_path):
    with pytest.raises(ValueError, match="case 8 is outside a campaign of 8 runs"):
        make_campaign(tmp_path, runs=8).case(8)


def test_an_unknown_plant_names_the_known_ones(tmp_path):
    campaign = Campaign("shuttle", seed=0, runs=1, duration_s=1.0, out_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown plant 'shuttle'; known plants: imu_spacecraft"):
        campaign.case(0)


def test_out_dir_is_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    campaign = Campaign("imu_spacecraft", seed=0, runs=1, duration_s=1.0, out_dir=Path("runs"))
    assert campaign.out_dir == tmp_path / "runs"
