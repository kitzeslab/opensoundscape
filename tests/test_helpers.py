from opensoundscape import helpers
from numpy import nan
import numpy as np
import pytest


def test_isnan():
    assert not helpers.isNan(0) and helpers.isNan(nan)


def test_sigmoid():
    helpers.sigmoid(-1)


def test_bound():
    assert helpers.bound(-1, [0, 1]) == 0 and helpers.bound(2, [0, 1]) == 1


def test_binarize():
    assert np.sum(helpers.binarize([-1, 1], 0)) == 1


def test_binarize_2d():
    assert np.sum(helpers.binarize([[0, 0.2], [5, 0.6]], 0.5)) == 2


def test_binarize_shape_error():
    with pytest.raises(ValueError):
        helpers.binarize([[[0, 0.2], [5, 0.6]]], 0.5)


def test_run_command():
    helpers.run_command("ls .")


def test_rescale_features():
    x = helpers.rescale_features([1, 2, 3], [1])
    assert x[0][0] == 1


def test_file_name():
    assert helpers.file_name("/abc/def/hij.kl") == "hij"


def test_hex_to_time():
    from datetime import datetime

    assert isinstance(helpers.hex_to_time("6EF223"), datetime)


def test_min_max_scale():
    scaled = helpers.min_max_scale([-5, 10.2], (0, 1))
    assert round(min(scaled)) == 0 and round(max(scaled)) == 1


def test_jitter():
    helpers.jitter([1, 2, 3], 1, distribution="gaussian")
    helpers.jitter([1, 2, 3], 1, distribution="uniform")


def test_jitter_nonexistant_raises_value_error():
    with pytest.raises(ValueError):
        helpers.jitter([1, 2, 3], 1, distribution="nonexistant")
