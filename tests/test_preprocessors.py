from copy import copy
from pathlib import Path

import pytest
import math
import numpy as np
import pandas as pd
import torch

from opensoundscape.preprocess import preprocessors, actions, action_functions
from opensoundscape.preprocess.preprocessors import (
    SpectrogramPreprocessor,
    AudioPreprocessor,
)
from opensoundscape.preprocess.utils import PreprocessingError
import warnings
from opensoundscape.audio import Audio
from opensoundscape.sample import AudioSample
from opensoundscape.utils import make_clip_df, set_seed


@pytest.fixture()
def sample():
    return AudioSample("tests/audio/silence_10s.mp3", labels=pd.Series({0: 0, 1: 1}))


@pytest.fixture()
def short_sample():
    return AudioSample("tests/audio/veryshort.wav", labels=pd.Series({0: 0, 1: 1}))


@pytest.fixture()
def preprocessor():
    return SpectrogramPreprocessor(sample_duration=2.0)


@pytest.fixture()
def preprocessor_3x50x50():
    return SpectrogramPreprocessor(sample_duration=2.0, height=50, width=50, channels=3)


@pytest.fixture()
def train_df():
    return pd.DataFrame(
        index=["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"],
        data=[[0, 1], [1, 0]],
    )


@pytest.fixture()
def temp_json_path(request):
    path = Path("tests/tmp_preprocessor.json")

    # remove this after tests are complete
    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def temp_yaml_path(request):
    path = Path("tests/tmp_preprocessor.yaml")

    # remove this after tests are complete
    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def train_df_clips(train_df):
    clip_df = make_clip_df(train_df.index.values, clip_duration=1.0)
    return clip_df


@pytest.fixture()
def preprocessor_with_overlay(train_df_clips):
    return SpectrogramPreprocessor(sample_duration=2.0, overlay_df=train_df_clips)


@pytest.fixture()
def audiopreprocessor():
    return preprocessors.AudioPreprocessor(2.0, sample_rate=22050)


def test_repr(preprocessor):
    print(preprocessor)


def test_generate_sample_makes_copy(preprocessor, sample):
    sample_copy = preprocessor._generate_sample(sample)
    assert sample_copy is not sample
    assert sample_copy.labels is not sample.labels


def test_init_overlay_preprocessor(preprocessor_with_overlay):
    pass


def test_spectrogram_preprocessor_output_size(
    preprocessor, preprocessor_3x50x50, preprocessor_with_overlay, sample
):
    """should return a sample with a tensor, check output shapes"""

    s = preprocessor.forward(sample)
    assert list(s.data.shape) == [1, 224, 224]
    s = preprocessor_3x50x50.forward(sample)
    assert list(s.data.shape) == [3, 50, 50]
    s = preprocessor_with_overlay.forward(sample)
    assert list(s.data.shape) == [1, 224, 224]
    # retain original spec height if width/height are None
    preprocessor.width = None
    preprocessor.height = None
    s = preprocessor.forward(sample)
    assert list(s.data.shape) == [1, 129, 343]


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
    # create the action object with a function and arguments
    action = (actions.Action(action_functions.tensor_add_noise, std=0.01),)
    # add it to the pipeline TWICE with different names as the keys
    preprocessor._insert_action_after("to_tensor", "add_noise_NEW", action)
    preprocessor._insert_action_before("add_noise_NEW", "new2", action)
    preprocessor.insert_action("new3", action, before_key="new2")
    preprocessor.insert_action("new4", action, after_key="new3")
    # should not allow trying to add an action with a name that already exists in the pipeline
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


def test_catch_input_to_forward_is_dataframe(preprocessor):
    # raises AssertionError if samples arg in forward is not pd.Series
    # which might occur if user accessed a dataset with a list instead of a single index
    # see issue 803: https://github.com/kitzeslab/opensoundscape/issues/803
    with pytest.raises(AssertionError):
        preprocessor.forward(pd.DataFrame({0: ["a"]}))


def test_catch_input_to_forward_is_dataframe(preprocessor):
    # raises AssertionError if samples arg in forward is not pd.Series
    # which might occur if user accessed a dataset with a list instead of a single index
    # see issue 803: https://github.com/kitzeslab/opensoundscape/issues/803
    with pytest.raises(AssertionError):
        preprocessor.forward(pd.DataFrame({0: ["a"]}))


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


def test_pcenpreprocessor(sample):

    p1 = preprocessors.PCENPreprocessor(sample_duration=1)
    s1 = p1.forward(sample, bypass_augmentations=True).data
    assert isinstance(s1, torch.Tensor)

    # try using some different settings
    p1.pipeline.pcen.set(gain=0.5)
    s2 = p1.forward(sample, bypass_augmentations=True).data


def test_preprocessor_to_from_dict(preprocessor, sample):
    set_seed(0)
    sample0 = preprocessor.forward(sample)
    d = preprocessor.to_dict()
    preprocessor2 = SpectrogramPreprocessor.from_dict(d)
    set_seed(0)
    sample2 = preprocessor2.forward(sample)
    assert np.array_equal(sample0.data, sample2.data)


def test_preprocessor_to_from_json(preprocessor, sample, temp_json_path):
    set_seed(0)
    sample0 = preprocessor.forward(sample)
    preprocessor.save(temp_json_path)
    preprocessor2 = preprocessors.load_json(temp_json_path)
    set_seed(0)
    sample2 = preprocessor2.forward(sample)
    assert np.array_equal(sample0.data, sample2.data)


def test_preprocessor_to_from_yaml(preprocessor, sample, temp_yaml_path):
    set_seed(0)
    sample0 = preprocessor.forward(sample)
    preprocessor.save(temp_yaml_path)
    preprocessor2 = preprocessors.load_yaml(temp_yaml_path)
    set_seed(0)
    sample2 = preprocessor2.forward(sample)
    assert np.array_equal(sample0.data, sample2.data)


def test_preprocessor_to_from_json_with_custom_action_fn(
    preprocessor, sample, temp_json_path
):
    @action_functions.register_action_fn
    def custom_gain(audio):
        return audio.apply_gain(dB=6)

    preprocessor.insert_action(
        action_index="custom_gain",
        action=actions.Action(fn=custom_gain),
        after_key="load_audio",
    )

    set_seed(0)
    sample0 = preprocessor.forward(sample)
    preprocessor.save(temp_json_path)

    preprocessor2 = preprocessors.load_json(temp_json_path)
    assert "custom_gain" in preprocessor2.pipeline
    assert preprocessor2.pipeline.custom_gain.action_fn == custom_gain

    set_seed(0)
    sample2 = preprocessor2.forward(sample)
    assert np.array_equal(sample0.data, sample2.data)


def test_preprocessor_to_from_json_with_custom_action_cls(
    preprocessor, sample, temp_json_path
):
    @actions.register_action_cls
    class CustomGain(actions.BaseAction):
        def __init__(self):
            super().__init__()
            self.gain_dB = 6

        def go(self, sample):
            sample.data = sample.data.apply_gain(dB=self.gain_dB)

    preprocessor.insert_action(
        action_index="custom_gain",
        action=CustomGain(),
        after_key="load_audio",
    )

    set_seed(0)
    sample0 = preprocessor.forward(sample)
    preprocessor.save(temp_json_path)

    preprocessor2 = preprocessors.load_json(temp_json_path)
    assert "custom_gain" in preprocessor2.pipeline
    assert type(preprocessor2.pipeline.custom_gain) == CustomGain

    set_seed(0)
    sample2 = preprocessor2.forward(sample)
    assert np.array_equal(sample0.data, sample2.data)


# several specific scenarios are tested using Datasets in test_datasets.py
