import pytest
from pathlib import Path
import numpy as np
import pandas as pd
from opensoundscape.audio import Audio
from numpy.testing import assert_allclose
from opensoundscape.preprocess import actions
from opensoundscape.preprocess.preprocessors import (
    CnnPreprocessor,
    AudioLoadingPreprocessor,
    PreprocessingError,
    LongAudioPreprocessor,
)
from PIL import Image
import warnings
import torch


@pytest.fixture()
def dataset_df():
    paths = ["tests/audio/silence_10s.mp3", "tests/audio/veryshort.wav"]
    labels = [[0, 1], [1, 0]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def overlay_df():
    paths = ["tests/audio/1min.wav", "tests/audio/great_plains_toad.wav"]
    labels = [[1, 0], [0, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def overlay_df_all_positive():
    paths = ["tests/audio/great_plains_toad.wav"]
    labels = [[1, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


def test_audio_loading_preprocessor(dataset_df):
    """should retain original sample rate"""
    dataset = AudioLoadingPreprocessor(dataset_df)
    assert dataset[0]["X"].samples.shape == (44100 * 10,)


def test_audio_resample(dataset_df):
    """should retain original sample rate"""
    dataset = AudioLoadingPreprocessor(dataset_df)
    dataset.actions.load_audio.set(sample_rate=16000)
    assert dataset[0]["X"].samples.shape == (16000 * 10,)


def test_cnn_preprocessor(dataset_df):
    """should return tensor and labels"""
    dataset = CnnPreprocessor(dataset_df)
    dataset.augmentation_off()
    sample1 = dataset[0]["X"]
    assert sample1.numpy().shape == (3, 224, 224)
    assert dataset[0]["y"].numpy().shape == (2,)


def test_cnn_preprocessor_augment_off(dataset_df):
    """should return same image each time"""
    dataset = CnnPreprocessor(dataset_df)
    dataset.augmentation_off()
    sample1 = dataset[0]["X"].numpy()
    sample2 = dataset[0]["X"].numpy()
    assert np.array_equal(sample1, sample2)


def test_cnn_preprocessor_augent_on(dataset_df):
    """should return different images each time"""
    dataset = CnnPreprocessor(dataset_df)
    sample1 = dataset[0]["X"]
    sample2 = dataset[0]["X"]
    assert not np.array_equal(sample1, sample2)


def test_cnn_preprocessor_overlay(dataset_df, overlay_df):
    dataset = CnnPreprocessor(dataset_df, overlay_df=overlay_df)
    sample1 = dataset[0]["X"]
    dataset.actions.overlay.off()
    sample2 = dataset[0]["X"]
    assert not np.array_equal(sample1, sample2)


def test_overlay_different_class(dataset_df, overlay_df):
    """just make sure it runs and doesn't hang"""
    dataset = CnnPreprocessor(dataset_df, overlay_df=overlay_df)
    dataset.actions.overlay.set(overlay_class="different")
    sample1 = dataset[0]["X"]


def test_overlay_different_class_warning(dataset_df, overlay_df_all_positive):
    """if no samples work, should give preprocessing error"""
    dataset = CnnPreprocessor(dataset_df, overlay_df=overlay_df_all_positive)
    dataset.actions.overlay.set(overlay_class="different")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sample1 = dataset[0]["X"]  # raises a warning
        assert len(w) == 1


def test_overlay_specific_class(dataset_df, overlay_df):
    """just make sure it runs and doesn't hang"""
    dataset = CnnPreprocessor(dataset_df, overlay_df=overlay_df)
    dataset.actions.overlay.set(overlay_class=0)
    sample1 = dataset[0]["X"]


def test_overlay_update_labels(dataset_df, overlay_df):
    """should return different images each time"""
    dataset = CnnPreprocessor(dataset_df, overlay_df=overlay_df)
    dataset.actions.overlay.set(overlay_class="different")
    dataset.actions.overlay.set(update_labels=True)
    sample = dataset[0]
    assert np.array_equal(sample["y"].numpy(), [1, 1])


def test_cnn_preprocessor_fails_on_short_file(dataset_df):
    """should fail on short file when audio duration is specified"""
    dataset = CnnPreprocessor(dataset_df, audio_length=5.0)
    with pytest.raises(PreprocessingError):
        sample = dataset[1]["X"]


def test_long_audio_dataset():
    df = pd.DataFrame(index=["tests/audio/1min.wav"])
    ds = LongAudioPreprocessor(
        df, audio_length=5.0, clip_overlap=0.0, out_shape=[224, 224]
    )
    superbatch = ds[0]
    assert superbatch["X"].shape == torch.Size([12, 3, 224, 224])


def test_long_audio_dataset_fails_on_short_audio():
    df = pd.DataFrame(index=["tests/audio/veryshort.wav"])
    ds = LongAudioPreprocessor(
        df, audio_length=5.0, clip_overlap=3, out_shape=[224, 224]
    )
    with pytest.raises(PreprocessingError):
        superbatch = ds[0]
