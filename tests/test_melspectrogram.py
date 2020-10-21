#!/usr/bin/env python3
from opensoundscape.audio import Audio
from opensoundscape.melspectrogram import MelSpectrogram
import pytest


@pytest.fixture()
def veryshort_wav_str():
    return "tests/veryshort.wav"


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
