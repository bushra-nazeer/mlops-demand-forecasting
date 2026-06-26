import numpy as np

from forecasting.monitor import psi


def test_psi_detects_distribution_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 4000)
    same = rng.normal(0, 1, 4000)
    shifted = rng.normal(3, 1, 4000)
    assert psi(a, same) < 0.1
    assert psi(a, shifted) > 0.2
