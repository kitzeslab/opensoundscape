import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from opensoundscape.audio import Audio
from numpy.testing import assert_allclose
from opensoundscape.preprocess import actions
from opensoundscape.preprocess.preprocessors import SpecPreprocessor, PreprocessingError
import warnings
import torch


@pytest.fixture()
def dataset_df():
    paths = ["tests/audio/silence_10s.mp3", "tests/audio/veryshort.wav"]
    labels = [[0, 1], [1, 0]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def overlay_df():
    paths = ["tests/audio/rugr_drum.wav", "tests/audio/stereo.wav"]
    labels = [[1, 0], [0, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def overlay_df_all_positive():
    paths = ["tests/audio/rugr_drum.wav"]
    labels = [[1, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


def test_interrupt_get_item(dataset_df):
    """should retain original sample rate"""
    dataset = SpecPreprocessor(dataset_df)
    audio = dataset.__getitem__(0, break_on_key="random_trim_audio")
    assert audio.samples.shape == (44100 * 10,)


def test_audio_resample(dataset_df):
    """should retain original sample rate"""
    dataset = SpecPreprocessor(dataset_df)
    dataset.pipeline.load_audio.set(sample_rate=16000)
    audio = dataset.__getitem__(0, break_on_key="random_trim_audio")
    assert audio.samples.shape == (16000 * 10,)


def test_spec_preprocessor(dataset_df):
    """should return tensor and labels"""
    dataset = SpecPreprocessor(dataset_df)
    dataset.augmentation_on = False
    sample1 = dataset[0]["X"]
    assert sample1.numpy().shape == (3, 224, 224)
    assert dataset[0]["y"].numpy().shape == (2,)


def test_cnn_preprocessor_augment_off(dataset_df):
    """should return same image each time"""
    dataset = SpecPreprocessor(dataset_df)
    dataset.augmentation_on = False
    sample1 = dataset[0]["X"].numpy()
    sample2 = dataset[0]["X"].numpy()
    assert np.array_equal(sample1, sample2)


def test_cnn_preprocessor_augent_on(dataset_df):
    """should return different images each time"""
    dataset = SpecPreprocessor(dataset_df)
    dataset.augmentation_on = True
    sample1 = dataset[0]["X"]
    sample2 = dataset[0]["X"]
    assert not np.array_equal(sample1, sample2)


def test_cnn_preprocessor_overlay(dataset_df, overlay_df):
    dataset = SpecPreprocessor(dataset_df, overlay_df=overlay_df)
    sample1 = dataset[0]["X"]
    dataset.pipeline.overlay.bypass = True
    sample2 = dataset[0]["X"]
    assert not np.array_equal(sample1, sample2)


def test_overlay_different_class(dataset_df, overlay_df):
    """just make sure it runs and doesn't hang"""
    dataset = SpecPreprocessor(dataset_df, overlay_df=overlay_df)
    dataset.pipeline.overlay.set(overlay_class="different")
    sample1 = dataset[0]["X"]


def test_overlay_different_class_warning(dataset_df, overlay_df_all_positive):
    """if no samples work, should give preprocessing error"""
    dataset = SpecPreprocessor(dataset_df, overlay_df=overlay_df_all_positive)
    dataset.pipeline.overlay.set(overlay_class="different")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sample1 = dataset[0]["X"]  # raises a warning
        assert len(w) == 1


def test_overlay_specific_class(dataset_df, overlay_df):
    """just make sure it runs and doesn't hang"""
    dataset = SpecPreprocessor(dataset_df, overlay_df=overlay_df)
    dataset.pipeline.overlay.set(overlay_class=0)
    sample1 = dataset[0]["X"]


def test_overlay_with_weight_range(dataset_df, overlay_df):
    """overlay should allow range [min,max] for overlay_weight"""
    dataset = SpecPreprocessor(dataset_df, overlay_df=overlay_df)
    dataset.pipeline.overlay.set(overlay_weight=[0.3, 0.7])
    sample1 = dataset[0]["X"]


def test_overlay_with_invalid_weight_range(dataset_df, overlay_df):
    """overlay should allow range [min,max] for overlay_weight"""
    dataset = SpecPreprocessor(dataset_df, overlay_df=overlay_df)
    with pytest.raises(PreprocessingError):
        dataset.pipeline.overlay.set(overlay_weight=[0.1, 1.1])
        sample1 = dataset[0]["X"]
    with pytest.raises(PreprocessingError):
        dataset.pipeline.overlay.set(overlay_weight=[0.1, 0.5, 0.9])
        sample1 = dataset[0]["X"]


def test_overlay_update_labels(dataset_df, overlay_df):
    """should return different images each time"""
    dataset = SpecPreprocessor(dataset_df, overlay_df=overlay_df)
    dataset.pipeline.overlay.set(overlay_class="different")
    dataset.pipeline.overlay.set(update_labels=True)
    sample = dataset[0]
    assert np.array_equal(sample["y"].numpy(), [1, 1])


def test_overlay_update_labels_duplicated_index(dataset_df, overlay_df):
    """duplicate indices of overlay_df are now removed, resolving
    a bug that caused duplicated indices to return 2-d labels.
    """
    dataset_df = pd.concat([dataset_df, dataset_df])
    dataset = SpecPreprocessor(dataset_df, overlay_df=overlay_df)
    dataset.pipeline.overlay.set(overlay_class="different")
    dataset.pipeline.overlay.set(update_labels=True)
    sample = dataset[0]
    assert np.array_equal(sample["y"].numpy(), [1, 1])


def test_cnn_preprocessor_fails_on_short_file(dataset_df):
    """should fail on short file when audio duration is specified"""
    dataset = SpecPreprocessor(dataset_df)
    dataset.pipeline.trim_audio.set(extend=False)
    with pytest.raises(PreprocessingError):
        sample = dataset[1]["X"]


def test_clips(dataset_df):
    import librosa
    from opensoundscape.helpers import generate_clip_times_df

    # prepare a df for clip loading preprocessor: start_time, end_time columns
    files = dataset_df.index.values
    clip_dfs = []
    for f in files:
        t = librosa.get_duration(filename=f)
        clips = generate_clip_times_df(t, 2, 0)
        clips.index = [f] * len(clips)
        clips.index.name = "file"
        clip_dfs.append(clips)
    clip_df = pd.concat(clip_dfs)
    dataset = SpecPreprocessor(dataset_df)
    dataset.clip_times_df = clip_df
    dataset.label_df = dataset.clip_times_df[[]]
    assert len(dataset) == 5

    # load a sample
    dataset[0]["X"]
