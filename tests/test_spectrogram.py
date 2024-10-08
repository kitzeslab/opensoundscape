#!/usr/bin/env python3
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram, MelSpectrogram
import pytest
import numpy as np
import math
import torch
from PIL.Image import Image


@pytest.fixture()
def veryshort_wav_str():
    return "tests/audio/veryshort.wav"


@pytest.fixture()
def cswa_str():
    return "tests/audio/aru_1.wav"


@pytest.fixture()
def spec():
    return Spectrogram(
        spectrogram=np.zeros((5, 10)),
        frequencies=np.linspace(0, 100, 5),
        times=np.linspace(0, 10, 10),
        window_samples=100,
        overlap_samples=50,
        window_type="Hann",
        audio_sample_rate=44100,
        scaling="spectrum",
    )


def test_spectrogram_raises_typeerror():
    with pytest.raises(TypeError):
        Spectrogram.from_audio("not samples")


def test_spectrogram_shape_of_veryshort(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    spec = Spectrogram.from_audio(audio, overlap_samples=384)
    assert spec.spectrogram.shape == (257, 21)
    assert spec.frequencies.shape == (257,)
    assert spec.times.shape == (21,)
    assert math.isclose(spec.window_length, 0.02321995465, abs_tol=1e-4)
    assert math.isclose(spec.window_step, 0.005804988662, abs_tol=1e-4)
    assert math.isclose(spec.duration, audio.duration, abs_tol=1e-2)
    assert math.isclose(spec.window_start_times[0], 0, abs_tol=1e-4)


def test_spectrogram_shape_of_windowlengths_overlapfraction(veryshort_wav_str):
    # test spectrogram construction using window length in s and overlap fraction
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    spec = Spectrogram.from_audio(
        audio, window_length_sec=0.02321995465, overlap_fraction=0.75
    )

    assert spec.spectrogram.shape == (257, 21)
    assert spec.frequencies.shape == (257,)
    assert spec.times.shape == (21,)


def test_construct_spectrogram_spectrogram_str_raises():
    with pytest.raises(TypeError):
        Spectrogram("raises", np.zeros((5)), np.zeros((10)), (-100, -20))


def test_construct_spectrogram_frequencies_str_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), "raises", np.zeros((10)), (-100, -20))


def test_construct_spectrogram_times_str_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((5)), "raises", (-100, -20))


def test_construct_spectrogram_spectrogram_1d_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5)), np.zeros((5)), np.zeros((5)), (-100, -20))


def test_construct_spectrogram_frequencies_2d_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5)), (-100, -20))


def test_construct_spectrogram_times_2d_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 5)), np.zeros((5)), np.zeros((5, 5)), (-100, -20))


def test_construct_spectrogram_dimensions_mismatch_raises_one():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((7)), (-100, -20))


def test_construct_spectrogram_dimensions_mismatch_raises_two():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((3)), np.zeros((10)), (-100, -20))


def test_construct_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10)), (-100, -20))


def test_bandpass_spectrogram(spec):
    spec = spec.bandpass(25, 75)
    assert np.allclose(spec.frequencies, np.array([25, 50, 75]))
    # make sure it didn't loose any properties
    assert spec.window_samples == 100
    assert spec.overlap_samples == 50
    assert spec.window_type == "Hann"
    assert spec.audio_sample_rate == 44100
    assert spec.scaling == "spectrum"


def test_bandpass_spectrogram_out_of_bounds(spec):
    """
    Test that bandpass raises ValueError when out_of_bounds_ok=False
    and the bandpass range is beyond the max value (100 Hz here)
    """
    with pytest.raises(ValueError):
        spec.bandpass(0, 110, out_of_bounds_ok=False)


def test_bandpass_spectrogram_not_out_of_bounds(spec):
    """should not raise error"""
    spec = spec.bandpass(0.0, 20.0, out_of_bounds_ok=False)
    assert spec.frequencies.max() < 30


def test_bandpass_spectrogram_bad_limits(spec):
    """should complain because low > high"""
    with pytest.raises(ValueError):
        spec.bandpass(4, 2)


def test_trim_spectrogram(spec):
    spec = spec.trim(2, 4)
    # make sure it didn't loose any properties
    assert spec.window_samples == 100
    assert spec.overlap_samples == 50
    assert spec.window_type == "Hann"
    assert spec.audio_sample_rate == 44100
    assert spec.scaling == "spectrum"


def test_limit_range():
    s = Spectrogram(
        np.random.normal(0, 200, [5, 10]), np.zeros((5)), np.zeros((10))
    ).limit_range(-100, -20)
    assert np.max(s.spectrogram) <= -20 and np.min(s.spectrogram) >= -100


def test_plot_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10)), (-100, -20)).plot()


def test_plot_spectrogram_kHz():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10)), (-100, -20)).plot(
        kHz=True
    )


def test_amplitude_spectrogram():
    Spectrogram(
        np.zeros((5, 10)), np.zeros((5)), np.zeros((10)), (-100, -20)
    ).amplitude()


def test_net_amplitude_spectrogram():
    Spectrogram(
        np.zeros((5, 10)), np.linspace(0, 100, 5), np.linspace(0, 10, 10), (-100, -20)
    ).net_amplitude([50, 100], [[0, 10], [20, 30]])


def test_to_image():
    assert isinstance(
        Spectrogram(
            np.zeros((5, 10)),
            np.linspace(0, 100, 5),
            np.linspace(0, 10, 10),
            (-100, -20),
        ).to_image(),
        Image,
    )


def test_to_image_with_bandpass():
    assert isinstance(
        Spectrogram(
            spectrogram=np.zeros((5, 10)),
            frequencies=np.linspace(0, 100, 5),
            times=np.linspace(0, 10, 10),
        ).to_image(),
        Image,
    )


def test_melspectrogram_underflow(cswa_str):
    """
    Fixed a bug where log transform was applied twice.
    Added a test to check the max value of dB scaled spec is as expected
    """
    audio = Audio.from_file(cswa_str)
    mel_spec = MelSpectrogram.from_audio(audio)
    assert math.isclose(mel_spec.spectrogram.max(), -30.914056301116943, abs_tol=1e-4)


def test_to_image_shape(spec):
    img = spec.to_image(shape=[10, 15], channels=2, return_type="torch")
    assert list(img.shape) == [2, 10, 15]  # channels, height, width

    # PIL
    img = spec.to_image(shape=[10, 15], channels=2, return_type="pil")
    assert img.size == (15, 10)

    # numpy
    img = spec.to_image(shape=[10, 15], channels=2, return_type="np")
    assert img.shape == (2, 10, 15)


def test_to_image_range(spec):
    """make sure spec is clipped to range 0-1 after the linear rescaling from range to 0-1"""
    img = spec.to_image(shape=[5, 6], channels=1, return_type="torch", range=(-20, -10))
    assert img.min() >= 0 and img.max() <= 1

    img = spec.to_image(shape=[5, 6], channels=1, return_type="torch", range=(5, 10))
    assert img.min() >= 0 and img.max() <= 1


def test_to_image_shape_None(spec):
    """should retain original shape of spectrogram if shape=None"""
    img = spec.to_image(shape=None, channels=2, return_type="torch")
    spec_shape = list(spec.spectrogram.shape)
    assert list(img.shape) == [2] + spec_shape  #  width

    # test when shape specifies only desired width
    img = spec.to_image(shape=[None, 6], channels=2, return_type="torch")
    assert list(img.shape) == [2] + [spec_shape[0]] + [6]

    # test when shape specifies only desired height
    img = spec.to_image(shape=[5, None], channels=2, return_type="torch")
    assert list(img.shape) == [2, 5] + [spec_shape[1]]


def test_to_image_colormap(spec):
    img = spec.to_image(
        shape=[5, 6], channels=3, return_type="torch", colormap="viridis"
    )
    assert img.shape == (3, 5, 6)

    # pil
    img = spec.to_image(shape=[5, 6], channels=3, return_type="pil", colormap="viridis")
    assert img.size == (6, 5)

    # numpy
    img = spec.to_image(shape=[5, 6], channels=3, return_type="np", colormap="viridis")
    assert img.shape == (3, 5, 6)


def test_melspectrogram_shape_of_S_for_veryshort(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    assert mel_spec.spectrogram.shape == (64, 11)


def test_melspectrogram_to_image_works(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    assert mel_spec.to_image()


def test_melspectrogram_to_image_numchannels(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    img = mel_spec.to_image(shape=(10, 20), channels=4)
    assert img.size == (20, 10)
    arr = np.array(img)
    assert arr.shape == (10, 20, 4)


def test_melspectrogram_to_image_alltypes(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    img = mel_spec.to_image(shape=(10, 20), return_type="pil")
    assert isinstance(img, Image)
    assert img.size == (20, 10)
    img = mel_spec.to_image(
        shape=(10, 20), return_type="pil", colormap="viridis", channels=3
    )
    assert img.size == (20, 10)
    img = mel_spec.to_image(shape=(10, 20), return_type="np")
    assert isinstance(img, np.ndarray)
    assert img.shape == (1, 10, 20)
    img = mel_spec.to_image(shape=(10, 20), return_type="torch")
    assert isinstance(img, torch.Tensor)
    assert img.shape == (1, 10, 20)


def test_melspectrogram_to_image_with_invert(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    positive = mel_spec.to_image(shape=(10, 20), invert=False, return_type="np")
    negative = mel_spec.to_image(shape=(10, 20), invert=True, return_type="np")
    assert np.allclose(negative, 1 - positive, 1e-4)


def test_melspectrogram_trim_works(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio).trim(0, 1)


def test_melspectrogram_bandpass_works(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio).bandpass(2000, 3000)
    assert mel_spec.spectrogram.shape == (8, 11)
