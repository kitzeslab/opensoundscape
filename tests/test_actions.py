import pytest
import math
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from opensoundscape.audio import Audio
from numpy.testing import assert_allclose
from opensoundscape.preprocess import actions, action_functions
from opensoundscape.sample import AudioSample
from PIL import Image
import torch
from opensoundscape.spectrogram import Spectrogram
from opensoundscape.preprocess.overlay import Overlay

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


@pytest.fixture()
def sample_df():
    return pd.DataFrame(
        index=["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"],
        data=[[0, 1], [1, 0]],
    )


## Tests ##


def test_audio_clip_loader_file(sample):
    action = actions.AudioClipLoader()
    action.__call__(sample)
    assert sample.data.sample_rate == 44100


def test_audio_clip_loader_resample(sample):
    action = actions.AudioClipLoader(sample_rate=32000)
    action.__call__(sample)
    assert sample.data.sample_rate == 32000


def test_audio_clip_loader_clip(sample_clip):
    action = actions.AudioClipLoader()
    action.__call__(sample_clip)
    assert math.isclose(sample_clip.data.duration, 2, abs_tol=1e-4)


def test_action_trim(sample_audio):
    action = actions.AudioTrim(target_duration=1)
    sample_audio.target_duration = 2  # should be ignored
    action.__call__(sample_audio)
    assert math.isclose(sample_audio.data.duration, 1.0, abs_tol=1e-4)


def test_action_random_trim(sample_audio):
    sample2 = copy.deepcopy(sample_audio)
    action = actions.AudioTrim(target_duration=0.001, random_trim=True)
    action.__call__(sample_audio)
    action.__call__(sample2)
    assert math.isclose(sample_audio.data.duration, 0.001, abs_tol=1e-4)
    # random trim should result in 2 different samples
    assert not math.isclose(sample_audio.start_time, sample2.start_time, abs_tol=1e-9)
    assert not np.array_equal(sample_audio.data.samples, sample2.data.samples)


def test_audio_trimmer_duration_None(sample_audio):
    """should not trim if target_duration=None"""
    action = actions.AudioTrim(target_duration=None)
    action.__call__(sample_audio)
    assert math.isclose(sample_audio.data.duration, 0.142086167800, abs_tol=1e-4)


def test_audio_trimmer_raises_error_on_short_clip(sample_audio):
    action = actions.AudioTrim(target_duration=10, extend=False)
    with pytest.raises(ValueError):
        action.__call__(sample_audio)


def test_audio_trimmer_extend_short_clip(sample_audio):
    action = actions.AudioTrim(target_duration=10)
    action.__call__(sample_audio)  # extend=True is default
    assert math.isclose(sample_audio.data.duration, 10, abs_tol=1e-4)


def test_audio_random_gain(sample_audio):
    # should reduce 10x if -20dB gain
    original_max = max(sample_audio.data.samples)
    action = actions.Action(action_functions.audio_random_gain, dB_range=[-20, -20])
    action.__call__(sample_audio)
    assert math.isclose(max(sample_audio.data.samples) * 10, original_max, abs_tol=1e-6)


def test_audio_add_noise(sample_audio):
    """smoke test: does it run?"""
    action = actions.Action(action_functions.audio_add_noise)
    action.__call__(sample_audio)
    action = actions.Action(
        action_functions.audio_add_noise, noise_dB=-100, signal_dB=10, color="pink"
    )
    action.__call__(sample_audio)


def test_spectrogram_to_tensor(sample, sample_audio):
    action = actions.SpectrogramToTensor()
    sample.data = Spectrogram.from_audio(sample_audio.data)
    # these attributes normally get set in SpectrogramPreprocessor._generate_sample
    sample.height = 20
    sample.width = 30
    sample.channels = 3

    action.__call__(sample)  # converts .data from Spectrogram to Tensor
    assert isinstance(sample.data, torch.Tensor)
    assert list(sample.data.shape) == [3, 20, 30]  # note channels as dim0


def test_spectrogram_to_tensor_range(sample, sample_audio):
    """ensure that range is limited to 0,1 and values are scaled correctly

    compare values of image to expected values, for both use_skimage=True and False

    use_skimage=True is the legacy behavior, and should be tested to ensure that it still
    produces the same values. use_skimage=False uses torch, is faster, and produces slightly
    different values
    """
    action = actions.SpectrogramToTensor(range=(-80, 0))

    # these attributes normally get set in SpectrogramPreprocessor._generate_sample
    sample.height = 20
    sample.width = 30
    sample.channels = 1

    # test default behavior using torch
    sample.data = Spectrogram.from_audio(sample_audio.data)
    action(sample)
    assert isinstance(sample.data, torch.Tensor)
    assert list(sample.data.shape) == [1, 20, 30]  # note channels as dim0
    assert math.isclose(sample.data.min(), 0.0, abs_tol=1e-6) and sample.data.max() < 1
    assert math.isclose(sample.data.mean(), 0.040718697011470795, abs_tol=1e-6)

    # and with lower db range
    sample.data = Spectrogram.from_audio(sample_audio.data)
    action.set(range=(-150, -90))
    action(sample)
    assert isinstance(sample.data, torch.Tensor)
    assert list(sample.data.shape) == [1, 20, 30]  # note channels as dim0
    assert sample.data.min() > 0 and math.isclose(sample.data.max(), 1.0, abs_tol=1e-6)
    assert math.isclose(sample.data.mean(), 0.8361802697181702, abs_tol=1e-6)

    # test matching legacy behavior with use_skimage=True
    action.set(use_skimage=True)
    action.set(range=(-80, 0))
    sample.data = Spectrogram.from_audio(sample_audio.data)
    action(sample)  # converts .data from Spectrogram to Tensor
    assert isinstance(sample.data, torch.Tensor)
    assert list(sample.data.shape) == [1, 20, 30]  # note channels as dim0
    assert math.isclose(sample.data.min(), 0.0, abs_tol=1e-6) and sample.data.max() < 1
    assert math.isclose(sample.data.mean(), 0.044159847293801575, abs_tol=1e-6)

    # repeat with lower range
    action.set(range=(-150, -90))
    sample.data = Spectrogram.from_audio(sample_audio.data)
    action(sample)  # converts .data from Spectrogram to Tensor
    assert sample.data.min() > 0 and math.isclose(sample.data.max(), 1.0, abs_tol=1e-6)
    assert math.isclose(sample.data.mean(), 0.8427285774873222, abs_tol=1e-6)


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
    action.__call__(sample)  # converts .data from Spectrogram to Tensor

    assert list(sample.data.shape) == [1] + spec_shape[0:2]  # note channels as dim0

    # repeat for just retaining height
    sample.data = spec
    sample.width = 19
    action.__call__(sample)  # converts .data from Spectrogram to Tensor
    assert list(sample.data.shape) == [1] + [spec_shape[0]] + [19]

    # repeat for just retaining width
    sample.data = spec
    sample.height = 21
    sample.width = None
    action.__call__(sample)  # converts .data from Spectrogram to Tensor
    assert list(sample.data.shape) == [1] + [21] + [spec_shape[1]]


def test_color_jitter(tensor):
    """test that color jitter changes the tensor so that channels differ"""
    tensor = action_functions.torch_color_jitter(tensor)
    assert not np.array_equal(tensor[0, :, :].numpy(), tensor[1, :, :].numpy())


def test_scale_tensor(tensor):
    """scale_tensor with 0,1 parameters should have no impact"""
    result = action_functions.scale_tensor(tensor, input_mean=0, input_std=1)
    assert np.array_equal(tensor.numpy(), result.numpy())


def test_generic_action(sample, tensor):
    """initialize the Action class with an arbitrary funciton and pass function args as kwargs

    the additional args become action.params Series
    """
    sample.data = tensor
    action = actions.Action(action_functions.scale_tensor, input_mean=0, input_std=2)
    action.__call__(sample)
    assert sample.data.max() * 2 == tensor.max()


def test_action_get_set():
    action = actions.Action(action_functions.scale_tensor, input_mean=0, input_std=2)
    assert action.get("input_std") == 2
    action.set(input_mean=1)
    assert action.params.get("input_mean") == 1


def test_modify_parameter_with_series_magic(tensor):
    action = actions.Action(action_functions.scale_tensor, input_mean=0, input_std=2)
    assert action.params["input_mean"] == 0
    assert action.params.input_mean == 0  # access with . syntax
    action.params.input_mean = 1  # set with . syntax
    assert action.params["input_mean"] == 1
    action.__call__(tensor)


def test_base_action_to_from_dict():
    action = actions.BaseAction(is_augmentation=True)
    d = action.to_dict()
    action2 = actions.BaseAction.from_dict(d)
    assert action2.is_augmentation == action.is_augmentation
    action3 = actions.action_from_dict(d)
    assert action3.is_augmentation == action.is_augmentation


def test_action_to_from_dict():
    action = actions.Action(action_functions.scale_tensor, input_mean=0, input_std=2)
    d = action.to_dict()
    action2 = actions.Action.from_dict(d)
    assert (action2.params.values == action.params.values).all()
    assert action2.action_fn == action.action_fn
    action3 = actions.action_from_dict(d)
    assert (action3.params.values == action.params.values).all()
    assert action3.action_fn == action.action_fn


def test_overlay_to_from_dict(sample_df):
    action = Overlay(overlay_df=sample_df, update_labels=True)
    d = action.to_dict()

    action2 = Overlay.from_dict(d)  # raises warning about not having overlay_df
    # new action will have empty overlay_df and will be bypassed
    assert action2.bypass == True
    assert action2.overlay_df.empty


def test_pcen(sample_audio):
    sample_audio.data = Spectrogram.from_audio(sample_audio.data, dB_scale=False)
    action = actions.Action(action_functions.pcen)
    original_spec = copy.copy(sample_audio.data.spectrogram)
    action(sample_audio)
    assert not np.array_equal(sample_audio.data.spectrogram, original_spec)
