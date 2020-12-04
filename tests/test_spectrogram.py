#!/usr/bin/env python3
from opensoundscape.audio import Audio
from opensoundscape.spectrogram import Spectrogram
import pytest
import numpy as np


@pytest.fixture()
def veryshort_wav_str():
    return "tests/veryshort.wav"


def test_spectrogram_raises_typeerror():
    with pytest.raises(TypeError):
        Spectrogram.from_audio("not samples")


def test_spectrogram_shape_of_veryshort(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    spec = Spectrogram.from_audio(audio, overlap_samples=384)
    assert spec.spectrogram.shape == (257, 21)
    assert spec.frequencies.shape == (257,)
    assert spec.times.shape == (21,)


def test_construct_spectrogram_spectrogram_str_raises():
    with pytest.raises(TypeError):
        Spectrogram("raises", np.zeros((5)), np.zeros((10)))


def test_construct_spectrogram_frequencies_str_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), "raises", np.zeros((10)))


def test_construct_spectrogram_times_str_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((5)), "raises")


def test_construct_spectrogram_spectrogram_1d_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5)), np.zeros((5)), np.zeros((5)))


def test_construct_spectrogram_frequencies_2d_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 5)), np.zeros((5, 5)), np.zeros((5)))


def test_construct_spectrogram_times_2d_raises():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 5)), np.zeros((5)), np.zeros((5, 5)))


def test_construct_spectrogram_dimensions_mismatch_raises_one():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((7)))


def test_construct_spectrogram_dimensions_mismatch_raises_two():
    with pytest.raises(TypeError):
        Spectrogram(np.zeros((5, 10)), np.zeros((3)), np.zeros((10)))


def test_construct_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10)))


def test_bandpass_spectrogram():
    Spectrogram(
        np.zeros((5, 10)), np.linspace(0, 100, 5), np.linspace(0, 10, 10)
    ).bandpass(2, 4)


def test_trim_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.linspace(0, 100, 5), np.linspace(0, 10, 10)).trim(
        2, 4
    )


def test_limit_db_range():
    s = Spectrogram(
        np.random.normal(0, 200, [5, 10]), np.zeros((5)), np.zeros((10))
    ).limit_db_range(-100, -20)
    assert np.max(s.spectrogram) <= -20 and np.min(s.spectrogram) >= -100


def test_plot_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10))).plot()


def test_amplitude_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10))).amplitude()


def test_net_amplitude_spectrogram():
    Spectrogram(
        np.zeros((5, 10)), np.linspace(0, 100, 5), np.linspace(0, 10, 10)
    ).net_amplitude([50, 100], [[0, 10], [20, 30]])


def test_to_image():
    from PIL.Image import Image

    print(
        type(
            Spectrogram(
                np.zeros((5, 10)), np.linspace(0, 100, 5), np.linspace(0, 10, 10)
            ).to_image()
        )
    )
    assert isinstance(
        Spectrogram(
            np.zeros((5, 10)), np.linspace(0, 100, 5), np.linspace(0, 10, 10)
        ).to_image(),
        Image,
    )
