#!/usr/bin/env python3
import opensoundscape as opso
from opensoundscape.audio import load_samples
from opensoundscape.spectrogram import Spectrogram
import pytest
import numpy as np


@pytest.fixture()
def veryshort_wav_str():
    return f"tests/veryshort.wav"


def test_spectrogram_raises_typeerror():
    with pytest.raises(TypeError):
        Spectrogram.from_audio("not samples")


def test_spectrogram_shape_of_veryshort(veryshort_wav_str):
    spec = Spectrogram.from_audio(load_samples(veryshort_wav_str))
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
