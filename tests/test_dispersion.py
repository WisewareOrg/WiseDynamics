"""`Dispersion` and `apply_draws`, and the IMU spacecraft's standard dispersion."""

import math

import numpy as np
import pytest

from wise_dynamics import Dispersion
from wise_dynamics.basilisk import ImuSpacecraft, ImuSpacecraftConfig
from wise_dynamics.dispersion import apply_draws

D2R = math.pi / 180.0


def test_standard_dispersion_has_the_specified_widths():
    """2000 draws per axis; each sample std within 10% of the specified 1-sigma (about 6 sigma
    of the estimate's own scatter) and each mean within 5 standard errors of nominal."""
    widths = {
        "sigma0_BN": 0.01,
        "omega0_BN_B": 0.2 * D2R,
        "gyro_bias_P": 1e-4,
        "accel_bias_P": 1e-4,
    }
    nominal = ImuSpacecraft.nominal()
    rng = np.random.default_rng(2026)
    draws = [ImuSpacecraft.standard_dispersion.draw(nominal, rng) for _ in range(2000)]

    assert list(ImuSpacecraft.standard_dispersion.std) == list(widths)
    for name, width in widths.items():
        values = np.array([draw[name] for draw in draws])
        offsets = values - np.array(getattr(nominal, name))
        assert np.all(np.abs(offsets.std(axis=0) - width) < 0.10 * width), name
        assert np.all(np.abs(offsets.mean(axis=0)) < 5 * width / math.sqrt(2000)), name


def test_apply_draws_replaces_the_named_fields_only():
    config = apply_draws(ImuSpacecraftConfig(), {"gyro_bias_P": np.array([1.0, 2.0, 3.0])})
    assert config.gyro_bias_P == (1.0, 2.0, 3.0)
    assert config == ImuSpacecraftConfig(gyro_bias_P=(1.0, 2.0, 3.0))


@pytest.mark.parametrize(
    ("draws", "message"),
    [
        ({"gyro_bias": np.zeros(3)}, "no field 'gyro_bias'"),
        ({"step_s": np.zeros(3)}, "not a 3-vector"),
        ({"gyro_bias_P": np.zeros(2)}, r"shape \(2,\)"),
    ],
)
def test_apply_draws_refuses_a_draw_that_fits_no_vector_field(draws, message):
    with pytest.raises(ValueError, match=message):
        apply_draws(ImuSpacecraftConfig(), draws)


def test_draw_refuses_an_unknown_field():
    with pytest.raises(ValueError, match="no field 'mass'"):
        Dispersion({"mass": 1.0}).draw(ImuSpacecraftConfig(), np.random.default_rng(0))
