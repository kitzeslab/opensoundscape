#!/usr/bin/env python3
import opensoundscape as opso
from opensoundscape.audio import load_samples
from opensoundscape.spectrogram import *
import pytest

@pytest.fixture()
def veryshort_wav_str():
    return f"tests/veryshort.wav"

def test_spectrogram_raises_typeerror():
    with pytest.raises(TypeError):
        Spectrogram("not samples")

def test_spectrogram_shape_of_veryshort(veryshort_wav_str):
    spec = Spectrogram(load_samples(veryshort_wav_str))
    assert spec.spectrogram.shape == (257, 21)
    assert spec.frequencies.shape == (257,)
    assert spec.times.shape == (21,)
