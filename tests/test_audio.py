#!/usr/bin/env python3
import opensoundscape as opso
from opensoundscape.audio import (
    Audio,
    OpsoLoadAudioInputError,
    OpsoLoadAudioInputTooLong,
)
import pytest
import pathlib
import io
import numpy as np
import os
from os.path import exists

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
def out_path():
    return f"tests/audio_tools_out"

@pytest.fixture()
def veryshort_wav_pathlib(veryshort_wav_str):
    return pathlib.Path(veryshort_wav_str)


@pytest.fixture()
def veryshort_wav_bytesio(veryshort_wav_str):
    with open(veryshort_wav_str, "rb") as f:
        return io.BytesIO(f.read())
    
@pytest.fixture()
def saved_wav():
    return f"tests/audio_out/saved.wav"

@pytest.fixture()
def saved_mp3():
    return f"tests/audio_out/saved.mp3"

def test_load_veryshort_wav_str_44100(veryshort_wav_str):
    s = Audio(veryshort_wav_str, sample_rate=44100)
    assert s.samples.shape == (6266,)


def test_load_veryshort_wav_str(veryshort_wav_str):
    s = Audio(veryshort_wav_str)
    assert s.samples.shape == (3133,)


def test_load_veryshort_wav_pathlib(veryshort_wav_pathlib):
    s = Audio(veryshort_wav_pathlib)
    assert s.samples.shape == (3133,)


def test_load_veryshort_wav_bytesio(veryshort_wav_bytesio):
    s = Audio(veryshort_wav_bytesio)
    assert s.samples.shape == (3133,)


def test_samples_class_is_immutable_samples(veryshort_wav_bytesio):
    with pytest.raises(AttributeError):
        s = Audio(veryshort_wav_bytesio)
        s.samples = None


def test_samples_class_is_immutable_sample_rate(veryshort_wav_bytesio):
    with pytest.raises(AttributeError):
        s = Audio(veryshort_wav_bytesio)
        s.sample_rate = None


def test_load_pathlib_and_bytesio_are_almost_equal(
    veryshort_wav_pathlib, veryshort_wav_bytesio
):
    s_pathlib = Audio(veryshort_wav_pathlib)
    s_bytesio = Audio(veryshort_wav_bytesio)
    np.testing.assert_allclose(s_pathlib.samples, s_bytesio.samples, atol=1e-7)


def test_load_silence_10s_mp3_str_asserts_too_long(silence_10s_mp3_str):
    with pytest.raises(OpsoLoadAudioInputTooLong):
        Audio(silence_10s_mp3_str, max_duration=5)


def test_load_not_a_file_asserts_not_a_file(not_a_file_str):
    with pytest.raises(FileNotFoundError):
        Audio(not_a_file_str)

def test_trim(silence_10s_mp3_str):
    s = Audio(silence_10s_mp3_str)
    assert(isinstance(s.trim(0,1),Audio))
    
def test_bandpass(silence_10s_mp3_str):
    s = Audio(silence_10s_mp3_str)
    assert(isinstance(s.bandpass(1,100),Audio))
    
def test_bandpass(silence_10s_mp3_str):
    s = Audio(silence_10s_mp3_str,sample_rate=10000)
    assert(isinstance(s.bandpass(.001,4999),Audio))
    
def test_bandpass_low_error(silence_10s_mp3_str):
    s = Audio(silence_10s_mp3_str)
    with pytest.raises(ValueError):
        s.bandpass(0,100)
        
def test_bandpass_high_error(silence_10s_mp3_str):
    s = Audio(silence_10s_mp3_str,sample_rate=10000)
    with pytest.raises(ValueError):
        s.bandpass(100,5000)
        
def test_spectrum(silence_10s_mp3_str):
    s = Audio(silence_10s_mp3_str)
    assert(len(s.spectrum())==2)

def test_save(silence_10s_mp3_str,saved_wav,out_path):
    if not exists(out_path):
        os.system(f'mkdir {outpath}')
    if exists(saved_wav):
        os.system(f'rm {saved_wav}')
    Audio(silence_10s_mp3_str).save(saved_wav)
    assert(exists(saved_wav))
    
def test_save_extension_error(silence_10s_mp3_str,saved_mp3):
    with pytest.raises(ValueError):
        Audio(silence_10s_mp3_str).save(saved_mp3)