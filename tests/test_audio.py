#!/usr/bin/env python3
import opensoundscape as opso
from opensoundscape.audio import *
import pytest
import pathlib
import io
import numpy as np


@pytest.fixture()
def veryshort_wav_str():
    return f"tests/veryshort.wav"


@pytest.fixture()
def silence_10s_mp3_str():
    return f"tests/silence_10s.mp3"


@pytest.fixture()
def not_a_file_str():
    return f"tests/not_a_file.wav"


@pytest.fixture()
def veryshort_wav_pathlib(veryshort_wav_str):
    return pathlib.Path(veryshort_wav_str)


@pytest.fixture()
def veryshort_wav_bytesio(veryshort_wav_str):
    with open(veryshort_wav_str, "rb") as f:
        return io.BytesIO(f.read())


def test_load_veryshort_wav_str_44100(veryshort_wav_str):
    s = load_samples(veryshort_wav_str, sample_rate=44100)
    assert s.samples.shape == (6266,)


def test_load_veryshort_wav_str(veryshort_wav_str):
    s = load_samples(veryshort_wav_str)
    assert s.samples.shape == (3133,)


def test_load_veryshort_wav_pathlib(veryshort_wav_pathlib):
    s = load_samples(veryshort_wav_pathlib)
    assert s.samples.shape == (3133,)


def test_load_veryshort_wav_bytesio(veryshort_wav_bytesio):
    s = load_samples(veryshort_wav_bytesio)
    assert s.samples.shape == (3133,)

def test_samples_class_is_immutable_samples(veryshort_wav_bytesio):
    with pytest.raises(AttributeError):
        s = load_samples(veryshort_wav_bytesio)
        s.samples = None

def test_samples_class_is_immutable_sample_rate(veryshort_wav_bytesio):
    with pytest.raises(AttributeError):
        s = load_samples(veryshort_wav_bytesio)
        s.sample_rate = None

def test_load_pathlib_and_bytesio_are_almost_equal(
    veryshort_wav_pathlib, veryshort_wav_bytesio
):
    s_pathlib = load_samples(veryshort_wav_pathlib)
    s_bytesio = load_samples(veryshort_wav_bytesio)
    np.testing.assert_allclose(s_pathlib.samples, s_bytesio.samples, atol=1e-7)


def test_load_silence_10s_mp3_str_asserts_too_long(silence_10s_mp3_str):
    with pytest.raises(OpsoLoadAudioInputTooLong):
        load_samples(silence_10s_mp3_str, max_duration=5)


def test_load_not_a_file_asserts_not_a_file(not_a_file_str):
    with pytest.raises(FileNotFoundError):
        load_samples(not_a_file_str)
