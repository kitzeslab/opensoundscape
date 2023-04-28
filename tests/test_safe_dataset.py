import pytest
import pandas as pd
import numpy as np
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.datasets import AudioFileDataset
from opensoundscape.ml.safe_dataset import SafeDataset


@pytest.fixture()
def preprocessor():
    pre = SpectrogramPreprocessor(sample_duration=5.0)
    return pre


@pytest.fixture()
def dataset(preprocessor):
    paths = ["tests/audio/veryshort.wav", "tests/audio/silence_10s.mp3"]
    labels = [[0, 1], [1, 0]]
    df = pd.DataFrame(index=paths, data=labels, columns=[0, 1])
    return AudioFileDataset(df, preprocessor)


def test_safe_dataset_handles_short_file(dataset):
    """should raise warning but not fail"""
    dataset = SafeDataset(dataset, invalid_sample_behavior="substitute")
    dataset.dataset.preprocessor.pipeline.trim_audio.set(extend=False)
    dataset.dataset.preprocessor.pipeline.random_trim_audio.set(extend=False)
    sample = dataset[0]

    # skips first sample when it fails and loads next
    assert np.array_equal(sample.labels.values, [1, 0])

    # stores failed samples in ._invalid_indices
    assert len(dataset._invalid_indices) == 1


def test_safe_dataset_returns_none(preprocessor):
    """should give None for the sample"""
    dataset = SafeDataset(preprocessor, invalid_sample_behavior="none")

    sample = dataset[0]

    # returns None for the sample
    assert sample is None

    # stores failed samples in ._invalid_indices
    assert len(dataset._invalid_indices) == 1


def test_safe_dataset_raises(preprocessor):
    """should raise an exception on bad sample"""
    dataset = SafeDataset(preprocessor, invalid_sample_behavior="raise")

    with pytest.raises(Exception):
        sample = dataset[0]
