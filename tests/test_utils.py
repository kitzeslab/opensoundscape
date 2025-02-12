import numpy as np
import pytest
import pandas as pd
import pytz
import datetime
import torch
import random
from opensoundscape.ml import cnn, cnn_architectures

from opensoundscape import utils


@pytest.fixture()
def silence_10s_mp3_str():
    return "tests/audio/silence_10s.mp3"


@pytest.fixture()
def metadata_wav_str():
    return "tests/audio/metadata.wav"


def test_isnan():
    assert not utils.isNan(0) and utils.isNan(np.nan)


def test_sigmoid():
    utils.sigmoid(-1)


def test_binarize():
    assert np.sum(utils.binarize([-1, 1], 0)) == 1


def test_binarize_2d():
    assert np.sum(utils.binarize([[0, 0.2], [5, 0.6]], 0.5)) == 2


def test_binarize_shape_error():
    with pytest.raises(ValueError):
        utils.binarize([[[0, 0.2], [5, 0.6]]], 0.5)


def test_rescale_features():
    x = utils.rescale_features([1, 2, 3], [1])
    assert x[0][0] == 1


def test_min_max_scale():
    scaled = utils.min_max_scale([-5, 10.2], (0, 1))
    assert round(min(scaled)) == 0 and round(max(scaled)) == 1


def test_jitter():
    utils.jitter([1, 2, 3], 1, distribution="gaussian")
    utils.jitter([1, 2, 3], 1, distribution="uniform")


def test_jitter_nonexistant_raises_value_error():
    with pytest.raises(ValueError):
        utils.jitter([1, 2, 3], 1, distribution="nonexistant")


def test_generate_clip_times_df_default():
    """many corner cases / alternatives are tested for audio.split()"""
    clip_df = utils.generate_clip_times_df(full_duration=10, clip_duration=5.0)
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 5.0
    assert clip_df.iloc[1]["end_time"] == 10.0


def test_floating_point_error():
    """accessing un-rounded float index causes KeyError"""
    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=0.2, rounding_precision=None
    )
    clip_df = clip_df.set_index("start_time")
    with pytest.raises(KeyError):
        clip_df.loc[0.6]


def test_rounding_avoids_fp_error():
    """default behavior rounds times to avoid key error"""
    clip_df = utils.generate_clip_times_df(
        full_duration=10,
        clip_duration=0.2,  # rounding_precision=10 default
    )
    clip_df = clip_df.set_index("start_time")
    clip_df.loc[0.6]


def test_generate_clip_times_df_extend():
    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=6.0, final_clip="extend"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 6.0
    assert clip_df.iloc[1]["start_time"] == 6.0
    assert clip_df.iloc[1]["end_time"] == 12.0


def test_generate_clip_times_df_remainder():
    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=6.0, final_clip="remainder"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 6.0
    assert clip_df.iloc[1]["start_time"] == 6.0
    assert clip_df.iloc[1]["end_time"] == 10.0


def test_generate_clip_times_df_full():
    clip_df = utils.generate_clip_times_df(
        full_duration=11, clip_duration=6.0, final_clip="full"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 6.0
    assert clip_df.iloc[1]["start_time"] == 5.0
    assert clip_df.iloc[1]["end_time"] == 11.0


def test_generate_clip_times_df_overlap():
    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=5, clip_overlap=2.5
    )
    assert clip_df.shape[0] == 3
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 2.5
    assert clip_df.iloc[1]["end_time"] == 7.5

    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=5, clip_overlap_fraction=0.5
    )
    assert clip_df.shape[0] == 3
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 2.5
    assert clip_df.iloc[1]["end_time"] == 7.5

    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=5, clip_step=2.5
    )
    assert clip_df.shape[0] == 3
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 2.5
    assert clip_df.iloc[1]["end_time"] == 7.5


def test_generate_clip_times_df_overlap_raises_overspecified():
    with pytest.raises(ValueError):
        utils.generate_clip_times_df(
            full_duration=10,
            clip_duration=5,
            clip_overlap=2.5,
            clip_overlap_fraction=0.5,
        )
    with pytest.raises(ValueError):
        utils.generate_clip_times_df(
            full_duration=10,
            clip_duration=5,
            clip_overlap=2.5,
            clip_step=0.5,
        )
    with pytest.raises(ValueError):
        utils.generate_clip_times_df(
            full_duration=10,
            clip_duration=5,
            clip_overlap_fraction=0.5,
            clip_step=0.5,
        )


def test_make_clip_df(silence_10s_mp3_str):
    """many corner cases / alternatives are tested for audio.split()

    by default, notafile.wav makes 1 row with nan as start_time and end_time
    (controlled by raise_exceptions argument)
    """
    clip_df, invalid_samples = utils.make_clip_df(
        files=[silence_10s_mp3_str, silence_10s_mp3_str, "notafile.wav"],
        clip_duration=5.0,
        return_invalid_samples=True,
    )
    assert len(clip_df) == 5
    assert len(invalid_samples) == 1


def test_make_clip_df_audio_root():
    """many corner cases / alternatives are tested for audio.split()

    by default, notafile.wav makes 1 row with nan as start_time and end_time
    (controlled by raise_exceptions argument)
    """
    clip_df, invalid_samples = utils.make_clip_df(
        files=["silence_10s.mp3"],
        clip_duration=5.0,
        return_invalid_samples=True,
        audio_root="tests/audio/",
    )
    assert len(clip_df) == 2


def test_make_clip_df_raise(silence_10s_mp3_str):
    """many corner cases / alternatives are tested for audio.split()"""
    with pytest.raises(utils.GetDurationError):
        clip_df, invalid_samples = utils.make_clip_df(
            files=[silence_10s_mp3_str, silence_10s_mp3_str, metadata_wav_str],
            clip_duration=5.0,
            return_invalid_samples=True,
            raise_exceptions=True,
        )


def test_make_clip_df_from_label_df(silence_10s_mp3_str, metadata_wav_str):
    label_df = pd.DataFrame(
        {"a": [0, 1, 2]},
        index=[silence_10s_mp3_str, silence_10s_mp3_str, metadata_wav_str],
    )
    clip_df = utils.make_clip_df(label_df, clip_duration=5.0)

    # should copy labels for each file to all clips of that file
    # duplicate file should have labels from _first_ occurrence in label_df
    assert np.array_equal(clip_df["a"].values, [0, 0, 0, 0, 2, 2])


# The @pytest.mark.parametrize decorator loops trough each value in list when running pytest.
# If you add --verbose, it also prints if it passed for each value in the list for each function
# that takes it as input.

# For all utils.set_seed() tests, assert that results are determistic for the the same seed AND
# for different seeds, in a tensor/array at least one element is different.


@pytest.mark.parametrize("input", [1, 11, 13, 42, 59, 666, 1234])
def test_torch_rand(input):
    utils.set_seed(input)
    tr1 = torch.rand(100)

    utils.set_seed(input)
    tr2 = torch.rand(100)

    utils.set_seed(input + 1)
    tr3 = torch.rand(100)

    assert all(tr1 == tr2) & any(tr1 != tr3)


@pytest.mark.parametrize("input", [1, 11, 13, 42, 59, 666, 1234])
def test_numpy_random_rand(input):
    utils.set_seed(input)
    nr1 = np.random.rand(100)

    utils.set_seed(input)
    nr2 = np.random.rand(100)

    utils.set_seed(input + 1)
    nr3 = np.random.rand(100)

    assert all(nr1 == nr2) & any(nr1 != nr3)


@pytest.mark.parametrize("input", [1, 11, 13, 42, 59, 666, 1234])
def test_radom_sample(input):
    list1000 = list(range(1, 1000))

    utils.set_seed(input)
    rs1 = random.sample(list1000, 100)

    utils.set_seed(input)
    rs2 = random.sample(list1000, 100)

    utils.set_seed(input + 1)
    rs3 = random.sample(list1000, 100)

    assert (rs1 == rs2) & (rs1 != rs3)


@pytest.mark.parametrize("input", [1, 11, 13, 42, 59, 666, 1234])
def test_set_seed_results_in_deterministic_weights_cnn_init(input):
    """initializing a CNN with random weights should be deterministic after running utils.set_seed()"""
    utils.set_seed(input)
    model_resnet1 = cnn_architectures.resnet18(num_classes=10, weights=None)
    lw1 = model_resnet1.layer1[0].conv1.weight

    utils.set_seed(input)
    model_resnet2 = cnn_architectures.resnet18(num_classes=10, weights=None)
    lw2 = model_resnet2.layer1[0].conv1.weight

    utils.set_seed(input + 1)
    model_resnet3 = cnn_architectures.resnet18(num_classes=10, weights=None)
    lw3 = model_resnet3.layer1[0].conv1.weight

    assert torch.all(lw1 == lw2) & torch.any(lw1 != lw3)


def test_cast_np_to_native():
    """test that np int and float are cast to native, other types unaffected"""
    assert isinstance(utils.cast_np_to_native(np.int32(1)), int)
    assert isinstance(utils.cast_np_to_native(np.float32(1.0)), float)
    # should not affect other dtypes
    assert isinstance(utils.cast_np_to_native(True), bool)
    assert isinstance(utils.cast_np_to_native("a"), str)
