import pytest
import numpy as np
import pandas as pd
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader


@pytest.fixture()
def dataset_df():
    paths = ["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"]
    labels = [[1, 0], [0, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def bad_dataset_df():
    labels = [[1, 0], [0, 1]]
    return pd.DataFrame(index=range(len(labels)), data=labels, columns=[0, 1])


@pytest.fixture()
def dataset_df_multiindex():
    paths = ["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"]
    start_times = [0, 1]
    end_times = [1, 2]
    return pd.DataFrame(
        {
            "file": paths,
            "start_time": start_times,
            "end_time": end_times,
            "A": [0, 1],
            "B": [1, 0],
        }
    ).set_index(["file", "start_time", "end_time"])


@pytest.fixture()
def bad_dataset_df_multiindex():
    paths = ["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"]
    start_times = [0, 1]
    end_times = [1, 2]
    return pd.DataFrame(
        {
            "file": paths,
            "start_time": start_times,
            "end_time": end_times,
            "A": [0, 1],
            "B": [1, 0],
        }
    )  # .set_index(["file", "start_time", "end_time"])


@pytest.fixture()
def bad_dataset_df():
    labels = [[1, 0], [0, 1]]
    return pd.DataFrame(index=range(len(labels)), data=labels, columns=[0, 1])


@pytest.fixture()
def pre():
    return SpectrogramPreprocessor(sample_duration=1)


def test_helpful_error_if_index_is_integer(bad_dataset_df, pre):
    with pytest.raises(AssertionError):
        SafeAudioDataloader(bad_dataset_df, pre)


def test_init(dataset_df, pre):
    SafeAudioDataloader(dataset_df, pre)


def test_init_multiindex(dataset_df, pre):
    SafeAudioDataloader(dataset_df, pre)


def test_catch_index_not_set(bad_dataset_df_multiindex, pre):
    with pytest.raises(AssertionError):
        SafeAudioDataloader(bad_dataset_df_multiindex, pre)
