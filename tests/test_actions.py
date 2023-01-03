import pytest
from math import isclose
from pathlib import Path
import numpy as np
import pandas as pd
from opensoundscape.audio import Audio
from numpy.testing import assert_allclose
from opensoundscape.preprocess import actions
from PIL import Image
import torch

## Fixtures: prepare objects that can be used by tests ##


@pytest.fixture()
def short_wav_path():
    return "tests/audio/veryshort.wav"


@pytest.fixture()
def audio_short():
    return Audio.from_file("tests/audio/veryshort.wav")


@pytest.fixture()
def audio_10s():
    return Audio.from_file("tests/audio/silence_10s.mp3")


@pytest.fixture()
def audio_10s_path():
    return "tests/audio/silence_10s.mp3"


@pytest.fixture()
def tensor():
    x = np.random.uniform(0, 1, [3, 10, 10])
    return torch.Tensor(x)


@pytest.fixture()
def img():
    x = np.random.uniform(0, 255, [10, 10, 3])
    return Image.fromarray(x, mode="RGB")


## Tests ##


def test_audio_clip_loader_file(short_wav_path):
    action = actions.AudioClipLoader()
    audio = action.go(short_wav_path, _start_time=None, _end_time=None)
    assert audio.sample_rate == 44100


def test_audio_clip_loader_resample(short_wav_path):
    action = actions.AudioClipLoader(sample_rate=32000)
    audio = action.go(short_wav_path, _start_time=None, _end_time=None)
    assert audio.sample_rate == 32000


def test_audio_clip_loader_clip(audio_10s_path):
    action = actions.AudioClipLoader()
    audio = action.go(audio_10s_path, _start_time=0, _end_time=2)
    assert isclose(audio.duration, 2, abs_tol=1e-4)


def test_action_trim(audio_short):
    action = actions.AudioTrim()
    audio = action.go(audio_short, _sample_duration=1.0)
    assert isclose(audio.duration, 1.0, abs_tol=1e-4)


def test_action_random_trim(audio_short):
    action = actions.AudioTrim(random_trim=True)
    a = action.go(audio_short, _sample_duration=0.01)
    a2 = action.go(audio_short, _sample_duration=0.01)
    assert isclose(a.duration, 0.01, abs_tol=1e-4)
    assert not np.array_equal(a.samples, a2.samples)


def test_audio_trimmer_default(audio_10s):
    """should not trim if no extra args"""
    action = actions.AudioTrim()
    audio = action.go(audio_10s, _sample_duration=None)
    assert isclose(audio.duration, 10, abs_tol=1e-4)


def test_audio_trimmer_raises_error_on_short_clip(audio_short):
    action = actions.AudioTrim()
    with pytest.raises(ValueError):
        audio = action.go(audio_short, _sample_duration=10, extend=False)


def test_audio_trimmer_extend_short_clip(audio_short):
    action = actions.AudioTrim()
    audio = action.go(audio_short, _sample_duration=1.0)  # extend=True is default
    assert isclose(audio.duration, 1.0, abs_tol=1e-4)


def test_color_jitter(tensor):
    """test that color jitter changes the tensor so that channels differ"""
    tensor = actions.torch_color_jitter(tensor)
    assert not np.array_equal(tensor[0, :, :].numpy(), tensor[1, :, :].numpy())


def test_scale_tensor(tensor):
    """scale_tensor with 0,1 parameters should have no impact"""
    result = actions.scale_tensor(tensor, input_mean=0, input_std=1)
    assert np.array_equal(tensor.numpy(), result.numpy())


def test_generic_action(tensor):
    """should be able to provide function to Action plus kwargs"""
    action = actions.Action(actions.scale_tensor, input_mean=0, input_std=2)
    result = action.go(tensor)
    assert result.max() * 2 == tensor.max()


def test_action_get_set():
    action = actions.Action(actions.scale_tensor, input_mean=0, input_std=2)
    assert action.get("input_std") == 2
    action.set(input_mean=1)
    assert action.params.get("input_mean") == 1


def test_unexpected_param_raises_error():
    with pytest.raises(AssertionError):
        actions.Action(actions.scale_tensor, not_a_param=0)
    with pytest.raises(AssertionError):
        action = actions.Action(actions.scale_tensor)
        action.set(not_a_param=0)


def test_extra_arg():
    def me(x, _add):
        return x + _add

    action = actions.Action(me, extra_args=["_add"])
    assert action.go(0, _add=1) == 1
    with pytest.raises(TypeError):
        """raises error if we don't pass the extra arg"""
        action.go(0)


def test_modify_parameter_with_series_magic(tensor):
    action = actions.Action(actions.scale_tensor, input_mean=0, input_std=2)
    assert action.params["input_mean"] == 0
    assert action.params.input_mean == 0  # access with . syntax
    action.params.input_mean = 1  # set with . syntax
    assert action.params["input_mean"] == 1
    action.go(tensor)


# others tested implicitly through preprocessor and cnn tests
