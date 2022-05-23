import pytest
import numpy as np
import pandas as pd
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.preprocess.utils import PreprocessingError
import warnings
from opensoundscape.audio import Audio


@pytest.fixture()
def row():
    paths = ["tests/audio/silence_10s.mp3"]
    labels = [[1, 0]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1]).iloc[0]


@pytest.fixture()
def short_file_row():
    paths = ["tests/audio/veryshort.wav"]
    labels = [[0, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1]).iloc[0]


@pytest.fixture()
def preprocessor():
    return SpectrogramPreprocessor(2.0)


def test_repr(preprocessor):
    print(preprocessor)


def test_remove_action(preprocessor):
    original_length = len(preprocessor.pipeline)
    preprocessor.remove_action("load_audio")
    assert len(preprocessor.pipeline) == original_length - 1
    assert "load_audio" not in preprocessor.pipeline


def test_interrupt_get_item(preprocessor, row):
    """should retain original sample rate"""
    audio = preprocessor.forward(row, break_on_key="random_trim_audio")[0]
    assert type(audio) == Audio
    assert audio.samples.shape == (44100 * 10,)


def test_audio_resample(preprocessor, row):
    """should retain original sample rate"""
    preprocessor.pipeline.load_audio.set(sample_rate=16000)
    audio = preprocessor.forward(row, break_on_key="random_trim_audio")[0]
    assert audio.samples.shape == (16000 * 10,)


def test_spec_preprocessor_fails_on_short_file(short_file_row, preprocessor):
    """should fail on short file when audio duration is specified"""
    preprocessor.pipeline.trim_audio.set(extend=False)
    with pytest.raises(PreprocessingError):
        # if augmenting, random_trim extends!
        preprocessor.forward(short_file_row, bypass_augmentations=True)


def test_insert_action(preprocessor):
    from opensoundscape.preprocess.actions import Action, tensor_add_noise

    action = (Action(tensor_add_noise, std=0.01),)  # the action object
    preprocessor._insert_action_after("to_img", "add_noise_NEW", action)
    preprocessor._insert_action_before("add_noise_NEW", "new2", action)
    preprocessor.insert_action("new3", action, before_key="new2")
    preprocessor.insert_action("new4", action, after_key="new3")
    with pytest.raises(AssertionError):  # duplicate name
        preprocessor.insert_action("new4", action)


# several specific scenarios are tested using DataSets in test_datasets.py
