#!/usr/bin/env python3
import opensoundscape as opso
from opensoundscape.audio import Audio
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
    audio = Audio(veryshort_wav_str)
    spec = Spectrogram.from_audio(audio)
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
    Spectrogram(np.zeros((5, 10)), np.linspace(0,100,5), np.linspace(0,10,10)).bandpass([2,4])
    
def test_trim_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.linspace(0,100,5), np.linspace(0,10,10)).trim([2,4])

def test_plot_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10))).plot()

def test_power_signal_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.zeros((5)), np.zeros((10))).power_signal()

def test_net_power_signal_spectrogram():
    Spectrogram(np.zeros((5, 10)), np.linspace(0,100,5), np.linspace(0,10,10)).net_power_signal([50,100],[[0,10],[20,30]])

