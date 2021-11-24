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
def tensor():
    x = np.random.uniform(0, 1, [3, 10, 10])
    return torch.Tensor(x)


@pytest.fixture()
def img():
    x = np.random.uniform(0, 255, [10, 10, 3])
    return Image.fromarray(x, mode="RGB")


## Tests ##


def test_audio_loader(short_wav_path):
    action = actions.AudioLoader()
    audio = action.go(short_wav_path)
    assert audio.sample_rate == 44100


def test_audio_loader_resample(short_wav_path):
    action = actions.AudioLoader(sample_rate=32000)
    audio = action.go(short_wav_path)
    assert audio.sample_rate == 32000


def test_audio_trimmer(audio_10s):
    action = actions.AudioTrimmer(audio_length=1.0)
    audio = action.go(audio_10s)
    assert isclose(audio.duration(), 1.0, abs_tol=1e-4)


def test_audio_trimmer_random_trim(audio_10s):
    action = actions.AudioTrimmer(audio_length=0.1, random_trim=True)
    audio = action.go(audio_10s)
    audio2 = action.go(audio_10s)
    assert isclose(audio.duration(), 0.1, abs_tol=1e-4)
    assert not np.array_equal(audio.samples, audio2.samples)


def test_audio_trimmer_default(audio_10s):
    action = actions.AudioTrimmer()
    audio = action.go(audio_10s)
    assert isclose(audio.duration(), 10, abs_tol=1e-4)


def test_audio_trimmer_raises_error_on_short_clip(audio_short):
    action = actions.AudioTrimmer(audio_length=1.0)
    with pytest.raises(ValueError):
        audio = action.go(audio_short)


def test_audio_trimmer_extend_short_clip(audio_short):
    action = actions.AudioTrimmer(audio_length=1.0, extend=True)
    audio = action.go(audio_short)
    assert isclose(audio.duration(), 1, abs_tol=1e-4)


# def test_save_tensor_to_disk(tensor):
# action = SaveTensorToDisk('.')


def test_color_jitter(tensor):
    """test that color jitter changes the tensor so that channels differ"""
    action = actions.TorchColorJitter()
    tensor = action.go(tensor)
    assert not np.array_equal(tensor[0, :, :].numpy(), tensor[1, :, :].numpy())


def test_img_to_tensor(img):
    """result should have 3 channels"""
    action = actions.ImgToTensor()
    result = action.go(img)
    assert type(result) == torch.Tensor
    assert result.shape[0] == 3


def test_img_to_tensor_grayscale(img):
    """result should have 1 channel"""
    action = actions.ImgToTensorGrayscale()
    result = action.go(img)
    assert type(result) == torch.Tensor
    assert result.shape[0] == 1


def test_tensor_torch_normalize(tensor):
    action = actions.TensorNormalize(mean=0, std=1)
    result = action.go(tensor)
    assert np.array_equal(tensor.numpy(), result.numpy())


# def test_time_warp

# def test_time_mask

# def test_frequency_mask

# def test_tensor_augment
