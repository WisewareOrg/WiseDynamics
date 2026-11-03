"""`draw_standard`, and the IMU spacecraft's standard dispersion."""

import math

import numpy as np
import pytest

from wise_dynamics import draw_standard
from wise_dynamics.basilisk import ImuSpacecraft

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
    nominal = ImuSpacecraft()
    rng = np.random.default_rng(2026)
    draws = [draw_standard(ImuSpacecraft, rng) for _ in range(2000)]

    assert list(ImuSpacecraft.standard_dispersion) == list(widths)
    for name, width in widths.items():
        values = np.array([draw[name] for draw in draws])
        offsets = values - np.array(getattr(nominal, name))
        assert np.all(np.abs(offsets.std(axis=0) - width) < 0.10 * width), name
        assert np.all(np.abs(offsets.mean(axis=0)) < 5 * width / math.sqrt(2000)), name


@pytest.mark.parametrize(
    ("dispersion", "message"),
    [
        ({"gyro_bias": 1e-4}, "no keyword 'gyro_bias'"),
        ({"mass_kg": 1.0}, "'mass_kg' default is not a 3-vector"),
    ],
)
def test_draw_standard_refuses_a_width_for_no_3_vector_keyword(monkeypatch, dispersion, message):
    monkeypatch.setattr(ImuSpacecraft, "standard_dispersion", dispersion)
    with pytest.raises(ValueError, match=message):
        draw_standard(ImuSpacecraft, np.random.default_rng(0))
