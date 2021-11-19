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
    import pytz

    t = helpers.hex_to_time("5F16A04E")
    assert t == datetime(2020, 7, 21, 7, 59, 10, tzinfo=pytz.utc)


def test_hex_to_time_convert_est():
    from datetime import datetime
    import pytz

    t = helpers.hex_to_time("5F16A04E")
    t = t.astimezone(pytz.timezone("US/Eastern"))
    f = pytz.timezone("US/Eastern").localize(datetime(2020, 7, 21, 3, 59, 10))
    assert t == f


def test_min_max_scale():
    scaled = helpers.min_max_scale([-5, 10.2], (0, 1))
    assert round(min(scaled)) == 0 and round(max(scaled)) == 1


def test_jitter():
    helpers.jitter([1, 2, 3], 1, distribution="gaussian")
    helpers.jitter([1, 2, 3], 1, distribution="uniform")


def test_jitter_nonexistant_raises_value_error():
    with pytest.raises(ValueError):
        helpers.jitter([1, 2, 3], 1, distribution="nonexistant")


def test_generate_clip_times_df_default():
    """many corner cases / alternatives are tested for audio.split()"""
    clip_df = helpers.generate_clip_times_df(full_duration=10, clip_duration=5.0)
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 5.0
    assert clip_df.iloc[1]["end_time"] == 10.0


def test_generate_clip_times_df_extend():
    clip_df = helpers.generate_clip_times_df(
        full_duration=10, clip_duration=6.0, final_clip="extend"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 6.0
    assert clip_df.iloc[1]["start_time"] == 6.0
    assert clip_df.iloc[1]["end_time"] == 12.0


def test_generate_clip_times_df_remainder():
    clip_df = helpers.generate_clip_times_df(
        full_duration=10, clip_duration=6.0, final_clip="remainder"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 6.0
    assert clip_df.iloc[1]["start_time"] == 6.0
    assert clip_df.iloc[1]["end_time"] == 10.0


def test_generate_clip_times_df_full():
    clip_df = helpers.generate_clip_times_df(
        full_duration=11, clip_duration=6.0, final_clip="full"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 6.0
    assert clip_df.iloc[1]["start_time"] == 5.0
    assert clip_df.iloc[1]["end_time"] == 11.0


def test_generate_clip_times_df_overlap():
    clip_df = helpers.generate_clip_times_df(
        full_duration=10, clip_duration=5, clip_overlap=2.5
    )
    assert clip_df.shape[0] == 3
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 2.5
    assert clip_df.iloc[1]["end_time"] == 7.5
