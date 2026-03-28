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
    sample_audio.start_time = 0
    sample2 = copy.deepcopy(sample_audio)
    action = actions.AudioTrim(target_duration=0.001, random_trim=True)
    action.__call__(sample_audio)
    action.__call__(sample2)
    assert math.isclose(sample_audio.data.duration, 0.001, abs_tol=1e-4)
    # assert not math.isclose(sample_audio.start_time, sample2.start_time, abs_tol=1e-9)
    # now retains original start/end times from the sample (treats as immutable)
    assert sample2.start_time == 0
    # random trim should result in 2 different samples
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
    assert math.isclose(sample.data.mean(), 0.10102162, abs_tol=1e-6)

    # and with lower db range: results in larger values
    sample.data = Spectrogram.from_audio(sample_audio.data)
    action.set(range=(-150, -90))
    action(sample)
    assert isinstance(sample.data, torch.Tensor)
    assert list(sample.data.shape) == [1, 20, 30]  # note channels as dim0
    assert sample.data.min() > 0 and math.isclose(sample.data.max(), 1.0, abs_tol=1e-6)
    assert math.isclose(sample.data.mean(), 0.987182, abs_tol=1e-6)


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
    sample_audio.data = Spectrogram.from_audio(sample_audio.data)
    action = actions.Action(action_functions.pcen)
    original_spec = copy.copy(sample_audio.data.spectrogram)
    action(sample_audio)
    assert not np.array_equal(sample_audio.data.spectrogram, original_spec)


def test_adaptive_random_gain_respects_min_output_level():
    """Test that adaptive_random_gain respects min_output_level constraint"""
    # Create moderately quiet audio where the constraint can be satisfied
    # by restricting the lower bound of the gain range
    quiet_audio = Audio.noise(duration=1, sample_rate=22050, dBFS=-30)

    # Apply adaptive gain multiple times and check all outputs meet minimum
    # Use a scenario where: audio ~= -30 dBFS, min_level = -40, gain_range = (-20, 0)
    # This means the lower bound will be restricted but upper bound (0) is sufficient
    min_level = -40
    for _ in range(10):
        result = action_functions.adaptive_random_gain(
            quiet_audio, gain_range=(-20, 0), min_output_level=min_level
        )
        # Allow small tolerance for floating point precision
        assert (
            result.dBFS >= min_level - 1
        ), f"Output {result.dBFS} is below minimum {min_level}"


def test_adaptive_random_gain_actually_changes_audio(sample_audio):
    """Test that adaptive_random_gain actually modifies the audio"""
    original_samples = sample_audio.data.samples.copy()

    result = action_functions.adaptive_random_gain(
        sample_audio.data, gain_range=(-20, -10), min_output_level=-50
    )

    # Audio should be different from original
    assert not np.array_equal(result.samples, original_samples)


def test_random_lowpass(sample_audio):
    """Test that random_lowpass actually applies a lowpass filter"""
    original_samples = sample_audio.data.samples.copy()

    result = action_functions.random_lowpass(
        sample_audio.data,
        cutoff_range=(1000, 5000),
        probability=1.0,
        order_range=(1, 2),
    )

    # Audio should be different from original (high freq content reduced)
    assert not np.array_equal(result.samples, original_samples)


def test_adaptive_random_gain_with_action_wrapper():
    """Test adaptive_random_gain works with Action wrapper"""
    audio = Audio.noise(duration=1, sample_rate=22050, dBFS=0)
    sample = AudioSample(None)
    sample.data = audio
    action = actions.Action(
        action_functions.adaptive_random_gain,
        gain_range=(-20, -10),
        min_output_level=-15,
    )
    original_level = audio.dBFS
    action(sample)

    # Audio level should have changed
    assert sample.data.dBFS < original_level


def test_adaptive_random_noise_adds_noise(sample_audio):
    """Test that adaptive_random_noise actually adds noise to audio"""
    original_samples = sample_audio.data.samples.copy()

    result = action_functions.adaptive_random_noise(
        sample_audio.data, snr_range=(-10, 0), signal_dB=0, color="white"
    )

    # Audio should be different from original (noise was added)
    assert not np.array_equal(result.samples, original_samples)


def test_adaptive_random_noise_adapts_to_signal_level():
    """Test that noise level adapts to input signal level"""
    # Create quiet and loud versions
    quiet_audio = Audio.noise(duration=1, sample_rate=22050, dBFS=-30)

    # Add noise to quiet audio
    snr_range = (2, 3)  # fairly narrow range for testing
    result = action_functions.adaptive_random_noise(
        quiet_audio, snr_range=snr_range, signal_dB=0, color="white"
    )
    assert quiet_audio.dBFS < result.dBFS < quiet_audio.dBFS + 6


def test_adaptive_random_noise_different_colors(sample_audio):
    """Test that adaptive_random_noise works with different noise colors"""
    colors = ["white", "pink", "brown"]

    for color in colors:
        result = action_functions.adaptive_random_noise(
            sample_audio.data, snr_range=(-10, 0), signal_dB=0, color=color
        )
        # Should produce different output
        assert not np.array_equal(result.samples, sample_audio.data.samples)


def test_adaptive_random_noise_stochastic(sample_audio):
    """Test that adaptive_random_noise produces different outputs on multiple calls"""
    results = []
    for _ in range(5):
        result = action_functions.adaptive_random_noise(
            sample_audio.data, snr_range=(-20, 0), signal_dB=0, color="white"
        )
        results.append(result.samples.copy())

    # All results should be different from each other
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            assert not np.array_equal(
                results[i], results[j]
            ), "Each call should produce different random noise"


def test_adaptive_random_noise_with_signal_gain(sample_audio):
    """Test that signal_dB parameter affects the output"""
    # Test with different signal_dB values
    result_0db = action_functions.adaptive_random_noise(
        sample_audio.data, snr_range=(-1, 1), signal_dB=0, color="white"
    )

    result_minus10db = action_functions.adaptive_random_noise(
        sample_audio.data, snr_range=(-1, 1), signal_dB=-10, color="white"
    )
    # Result with -10 dB signal gain should be quieter
    assert result_minus10db.dBFS < result_0db.dBFS


def test_adaptive_random_noise_with_action_wrapper(sample_audio):
    """Test adaptive_random_noise works with Action wrapper"""
    action = actions.Action(
        action_functions.adaptive_random_noise,
        snr_range=(-15, -5),
        signal_dB=0,
        color="pink",
    )
    original_samples = sample_audio.data.samples.copy()
    action(sample_audio)

    # Audio should be modified
    assert not np.array_equal(sample_audio.data.samples, original_samples)
