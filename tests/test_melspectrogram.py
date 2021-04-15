#!/usr/bin/env python3
from opensoundscape.audio import Audio
from opensoundscape.melspectrogram import MelSpectrogram
import pytest
import numpy as np


@pytest.fixture()
def veryshort_wav_str():
    return "tests/audio/veryshort.wav"


def test_melspectrogram_shape_of_S_for_veryshort(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    assert mel_spec.S.shape == (128, 98)


def test_melspectrogram_shape_of_S_with_pcen_maintains_shape(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    pcen_spec = mel_spec.to_pcen()
    assert mel_spec.S.shape == pcen_spec.S.shape


def test_melspectrogram_to_image_works(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    assert mel_spec.to_image()


def test_melspectrogram_to_image_with_reshape(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    img = mel_spec.to_image(shape=(10, 20))
    assert img.size == (10, 20)
    arr = np.array(img)
    assert arr.shape == (20, 10, 3)


def test_melspectrogram_to_image_with_mode(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    mel_spec = MelSpectrogram.from_audio(audio)
    img = mel_spec.to_image(shape=(10, 20), mode="L")
    arr = np.array(img)
    assert arr.shape == (20, 10)
