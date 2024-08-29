import pytest
import math
import numpy as np
import pandas as pd
from opensoundscape.preprocess import preprocessors
from opensoundscape.preprocess.utils import PreprocessingError
import warnings
from opensoundscape.audio import Audio
from opensoundscape.sample import AudioSample

from copy import copy


@pytest.fixture()
def sample():
    return AudioSample("tests/audio/silence_10s.mp3", labels=pd.Series({0: 0, 1: 1}))


@pytest.fixture()
def short_sample():
    return AudioSample("tests/audio/veryshort.wav", labels=pd.Series({0: 0, 1: 1}))


@pytest.fixture()
def preprocessor():
    return preprocessors.SpectrogramPreprocessor(2.0)


@pytest.fixture()
def audiopreprocessor():
    return preprocessors.AudioPreprocessor(2.0, sample_rate=22050)


def test_repr(preprocessor):
    print(preprocessor)


def test_generate_sample_makes_copy(preprocessor, sample):
    sample_copy = preprocessor._generate_sample(sample)
    assert sample_copy is not sample
    assert sample_copy.labels is not sample.labels


def test_remove_action(preprocessor):
    original_length = len(preprocessor.pipeline)
    preprocessor.remove_action("load_audio")
    assert len(preprocessor.pipeline) == original_length - 1
    assert "load_audio" not in preprocessor.pipeline


def test_interrupt_get_item(preprocessor, sample):
    """should retain original sample rate"""
    audio = preprocessor.forward(sample, break_on_key="random_trim_audio").data
    assert type(audio) == Audio
    assert audio.samples.shape == (44100 * 10,)


def test_profile(preprocessor, sample):
    """sample should have .runtime attribute with index matching preprocessor.pipeline, and float values"""
    sample = preprocessor.forward(sample, profile=True)
    # should report the time to load the audio
    assert sample.runtime[preprocessor.pipeline.index.values[0]] > 0
    assert (sample.runtime.index == preprocessor.pipeline.index).all()


def test_audio_resample(preprocessor, sample):
    """should retain original sample rate"""
    preprocessor.pipeline.load_audio.set(sample_rate=16000)
    audio = preprocessor.forward(sample, break_on_key="random_trim_audio").data
    assert audio.samples.shape == (16000 * 10,)


def test_spec_preprocessor_fails_on_short_file(short_sample, preprocessor):
    """should fail on short file when audio duration is specified"""
    preprocessor.pipeline.trim_audio.set(extend=False)
    with pytest.raises(PreprocessingError):
        # if augmenting, random_trim extends!
        preprocessor.forward(short_sample, bypass_augmentations=True)


def test_insert_action(preprocessor):
    from opensoundscape.preprocess.actions import Action, tensor_add_noise

    action = (Action(tensor_add_noise, std=0.01),)  # the action object
    preprocessor._insert_action_after("to_tensor", "add_noise_NEW", action)
    preprocessor._insert_action_before("add_noise_NEW", "new2", action)
    preprocessor.insert_action("new3", action, before_key="new2")
    preprocessor.insert_action("new4", action, after_key="new3")
    with pytest.raises(AssertionError):  # duplicate name
        preprocessor.insert_action("new4", action)


def test_trace_off(preprocessor, sample):
    sample = preprocessor.forward(sample)
    assert sample.trace is None


def test_trace_on(preprocessor, sample):
    sample = preprocessor.forward(sample, trace=True)
    # check that the saved values in the _trace match the expected
    # type returned by an action:
    assert isinstance(sample.trace["load_audio"], Audio)


def test_trace_output(preprocessor, sample):
    sample = preprocessor.forward(sample, trace=True)
    assert isinstance(sample.trace["load_audio"], Audio)


def test_audiopreprocessor(audiopreprocessor, sample):
    """should retain original sample rate"""
    s = audiopreprocessor.forward(sample).data
    assert type(s) == Audio
    assert math.isclose(s.duration, 2.0, abs_tol=1e-9)
    assert s.sample_rate == 22050


def test_audiopreprocessor_extend(audiopreprocessor, short_sample):
    """should retain original sample rate"""
    s = audiopreprocessor.forward(short_sample).data
    assert type(s) == Audio
    assert math.isclose(s.duration, 2.0, abs_tol=1e-9)
    assert s.sample_rate == 22050

    # when trim_audio.extend is False, should raise an error
    # if the input is too short
    audiopreprocessor.pipeline.trim_audio.set(extend=False)
    with pytest.raises(PreprocessingError):
        s = audiopreprocessor.forward(short_sample).data


def test_noisereduceaudiopreprocessor(sample):
    p1 = preprocessors.NoiseReduceAudioPreprocessor(
        sample_duration=1, sample_rate=16000, noisereduce_kwargs=dict(prop_decrease=1)
    )
    p2 = preprocessors.NoiseReduceAudioPreprocessor(
        sample_duration=1, sample_rate=16000, noisereduce_kwargs=dict(prop_decrease=0.5)
    )
    s1 = p1.forward(sample, bypass_augmentations=True).data
    s2 = p2.forward(sample, bypass_augmentations=True).data
    assert s1.rms < s2.rms


def test_noisereducespectrogrampreprocessor(short_sample):
    p1 = preprocessors.NoiseReduceSpectrogramPreprocessor(
        sample_duration=1, noisereduce_kwargs=dict(prop_decrease=1)
    )
    p2 = preprocessors.NoiseReduceSpectrogramPreprocessor(
        sample_duration=1, noisereduce_kwargs=dict(prop_decrease=0.5)
    )
    s1 = p1.forward(short_sample, bypass_augmentations=True).data
    s2 = p2.forward(short_sample, bypass_augmentations=True).data
    assert s1.mean() < s2.mean()


# several specific scenarios are tested using DataSets in test_datasets.py
