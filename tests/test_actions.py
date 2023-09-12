import pytest
import math
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from opensoundscape.audio import Audio
from numpy.testing import assert_allclose
from opensoundscape.preprocess import actions
from opensoundscape.sample import AudioSample
from PIL import Image
import torch
from opensoundscape.spectrogram import Spectrogram

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
def sample(short_wav_path):
    s = AudioSample(short_wav_path, start_time=None, duration=None)
    return s


@pytest.fixture()
def sample_audio(short_wav_path, audio_short):
    s = AudioSample(short_wav_path, start_time=None, duration=None)
    s.data = audio_short
    return s


@pytest.fixture()
def sample_clip(audio_10s_path):
    s = AudioSample(audio_10s_path, start_time=0, duration=2)
    return s


@pytest.fixture()
def img():
    x = np.random.uniform(0, 255, [10, 10, 3])
    return Image.fromarray(x, mode="RGB")


## Tests ##


def test_audio_clip_loader_file(sample):
    action = actions.AudioClipLoader()
    action.go(sample)
    assert sample.data.sample_rate == 44100


def test_audio_clip_loader_resample(sample):
    action = actions.AudioClipLoader(sample_rate=32000)
    action.go(sample)
    assert sample.data.sample_rate == 32000


def test_audio_clip_loader_clip(sample_clip):
    action = actions.AudioClipLoader()
    action.go(sample_clip)
    assert math.isclose(sample_clip.data.duration, 2, abs_tol=1e-4)


def test_action_trim(sample_audio):
    action = actions.AudioTrim()
    sample_audio.target_duration = 1.0
    action.go(sample_audio)
    assert math.isclose(sample_audio.data.duration, 1.0, abs_tol=1e-4)


def test_action_random_trim(sample_audio):
    sample2 = copy.deepcopy(sample_audio)
    action = actions.AudioTrim(random_trim=True)
    original_duration = sample_audio.data.duration
    sample_audio.target_duration = sample2.target_duration = 0.01
    action.go(sample_audio)
    action.go(sample2)
    assert math.isclose(sample_audio.data.duration, 0.01, abs_tol=1e-4)
    # random trim should result in 2 different samples
    assert not np.array_equal(sample_audio.data.samples, sample2.data.samples)


def test_audio_trimmer_default(sample_audio):
    """should not trim if no extra args"""
    action = actions.AudioTrim()
    sample_audio.target_duration = None
    action.go(sample_audio)
    assert math.isclose(sample_audio.data.duration, 0.142086167800, abs_tol=1e-4)


def test_audio_trimmer_raises_error_on_short_clip(sample_audio):
    action = actions.AudioTrim()
    sample_audio.target_duration = 10
    with pytest.raises(ValueError):
        action.go(sample_audio, extend=False)


def test_audio_trimmer_extend_short_clip(sample_audio):
    action = actions.AudioTrim()
    sample_audio.target_duration = 1
    action.go(sample_audio)  # extend=True is default
    assert math.isclose(sample_audio.data.duration, 1.0, abs_tol=1e-4)


def test_audio_random_gain(sample_audio):
    # should reduce 10x if -20dB gain
    original_max = max(sample_audio.data.samples)
    action = actions.Action(actions.audio_random_gain, dB_range=[-20, -20])
    action.go(sample_audio)
    assert math.isclose(max(sample_audio.data.samples) * 10, original_max, abs_tol=1e-6)


def test_audio_add_noise(sample_audio):
    """smoke test: does it run?"""
    action = actions.Action(actions.audio_add_noise)
    action.go(sample_audio)
    action = actions.Action(
        actions.audio_add_noise, noise_dB=-100, signal_dB=10, color="pink"
    )
    action.go(sample_audio)


def test_spectrogram_to_tensor(sample, sample_audio):
    action = actions.SpectrogramToTensor()
    sample.data = Spectrogram.from_audio(sample_audio.data)
    # these attributes normally get set in SpectrogramPreprocessor._generate_sample
    sample.height = 20
    sample.width = 30
    sample.channels = 3

    action.go(sample)  # converts .data from Spectrogram to Tensor
    assert isinstance(sample.data, torch.Tensor)
    assert list(sample.data.shape) == [3, 20, 30]  # note channels as dim0


def test_spectrogram_to_tensor_retain_shape(sample, sample_audio):
    """
    test that SpectrogramToTensor retains the shape of the spectrogram
    if no shape is provided
    """
    action = actions.SpectrogramToTensor()
    spec = Spectrogram.from_audio(sample_audio.data)
    sample.data = spec
    spec_shape = list(spec.spectrogram.shape)

    # these attributes normally get set in SpectrogramPreprocessor._generate_sample
    sample.height = None
    sample.width = None
    sample.channels = 1
    action.go(sample)  # converts .data from Spectrogram to Tensor

    assert list(sample.data.shape) == [1] + spec_shape[0:2]  # note channels as dim0

    # repeat for just retaining height
    sample.data = spec
    sample.width = 19
    action.go(sample)  # converts .data from Spectrogram to Tensor
    assert list(sample.data.shape) == [1] + [spec_shape[0]] + [19]

    # repeat for just retaining width
    sample.data = spec
    sample.height = 21
    sample.width = None
    action.go(sample)  # converts .data from Spectrogram to Tensor
    assert list(sample.data.shape) == [1] + [21] + [spec_shape[1]]


def test_color_jitter(tensor):
    """test that color jitter changes the tensor so that channels differ"""
    tensor = actions.torch_color_jitter(tensor)
    assert not np.array_equal(tensor[0, :, :].numpy(), tensor[1, :, :].numpy())


def test_scale_tensor(tensor):
    """scale_tensor with 0,1 parameters should have no impact"""
    result = actions.scale_tensor(tensor, input_mean=0, input_std=1)
    assert np.array_equal(tensor.numpy(), result.numpy())


def test_generic_action(sample, tensor):
    """should be able to provide function to Action plus kwargs"""
    sample.data = tensor
    action = actions.Action(actions.scale_tensor, input_mean=0, input_std=2)
    action.go(sample)
    assert sample.data.max() * 2 == tensor.max()


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


def test_modify_parameter_with_series_magic(tensor):
    action = actions.Action(actions.scale_tensor, input_mean=0, input_std=2)
    assert action.params["input_mean"] == 0
    assert action.params.input_mean == 0  # access with . syntax
    action.params.input_mean = 1  # set with . syntax
    assert action.params["input_mean"] == 1
    action.go(tensor)
