import pytest
import pandas as pd
import numpy as np
from opensoundscape.preprocess.preprocessors import CnnPreprocessor
from opensoundscape.torch.safe_dataset import SafeDataset


@pytest.fixture()
def preprocessor():
    paths = ["tests/audio/veryshort.wav", "tests/audio/silence_10s.mp3"]
    labels = [[0, 1], [1, 0]]
    df = pd.DataFrame(index=paths, data=labels, columns=[0, 1])
    return CnnPreprocessor(df, audio_length=5.0)


def test_safe_preprocessor_handles_short_file(preprocessor):
    """should raise warning but not fail"""
    dataset = SafeDataset(preprocessor, unsafe_behavior="substitute")

    sample = dataset[0]

    # skips first sample when it fails and loads next
    assert np.array_equal(sample["y"].numpy(), [1, 0])
    # stores failed samples in ._unsafe_indices
    assert len(dataset._unsafe_indices) == 1


def test_safe_preprocessor_returns_none(preprocessor):
    """should give None for the sample"""
    dataset = SafeDataset(preprocessor, unsafe_behavior="none")

    sample = dataset[0]

    # returns None for the sample
    assert sample is None

    # stores failed samples in ._unsafe_indices
    assert len(dataset._unsafe_indices) == 1


def test_safe_preprocessor_raises(preprocessor):
    """should raise an exception on bad sample"""
    dataset = SafeDataset(preprocessor, unsafe_behavior="raise")

    with pytest.raises(Exception):
        sample = dataset[0]