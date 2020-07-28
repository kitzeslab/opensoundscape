#!/usr/bin/env python3
import opensoundscape as opso
from opensoundscape.audio import (
    Audio,
    OpsoLoadAudioInputError,
    OpsoLoadAudioInputTooLong,
)
import pytest
from pathlib import Path
import io
import numpy as np
from random import uniform
from math import isclose


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
    return f"tests/audio_out"


@pytest.fixture()
def veryshort_wav_pathlib(veryshort_wav_str):
    return Path(veryshort_wav_str)


@pytest.fixture()
def veryshort_wav_bytesio(veryshort_wav_str):
    with open(veryshort_wav_str, "rb") as f:
        return io.BytesIO(f.read())


@pytest.fixture()
def tmp_dir(request):
    path = Path("tests/audio_out")
    if not path.exists():
        path.mkdir()

    def fin():
        path.rmdir()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def saved_wav(request, tmp_dir):
    path = Path(f"{tmp_dir}/saved.wav")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def saved_mp3(request, tmp_dir):
    path = Path(f"{tmp_dir}/saved.mp3")

    def fin():
        if path.exists():
            path.unlink()

    request.addfinalizer(fin)
    return path


def test_load_veryshort_wav_str_44100(veryshort_wav_str):
    s = Audio.from_file(veryshort_wav_str)
    assert s.samples.shape == (6266,)


def test_load_veryshort_wav_str(veryshort_wav_str):
    s = Audio.from_file(veryshort_wav_str, sample_rate=22050)
    assert s.samples.shape == (3133,)


def test_load_veryshort_wav_pathlib(veryshort_wav_pathlib):
    s = Audio.from_file(veryshort_wav_pathlib, sample_rate=22050)
    assert s.samples.shape == (3133,)


def test_load_veryshort_wav_bytesio(veryshort_wav_bytesio):
    s = Audio.from_bytesio(veryshort_wav_bytesio, sample_rate=22050)
    assert s.samples.shape == (3133,)


def test_load_pathlib_and_bytesio_are_almost_equal(
    veryshort_wav_pathlib, veryshort_wav_bytesio
):
    s_pathlib = Audio.from_file(veryshort_wav_pathlib)
    s_bytesio = Audio.from_bytesio(veryshort_wav_bytesio)
    np.testing.assert_allclose(s_pathlib.samples, s_bytesio.samples, atol=1e-7)


def test_load_silence_10s_mp3_str_asserts_too_long(silence_10s_mp3_str):
    with pytest.raises(OpsoLoadAudioInputTooLong):
        Audio.from_file(silence_10s_mp3_str, max_duration=5)


def test_load_not_a_file_asserts_not_a_file(not_a_file_str):
    with pytest.raises(FileNotFoundError):
        Audio.from_file(not_a_file_str)


def test_property_trim_length_is_correct(silence_10s_mp3_str):
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    duration = audio.duration()
    for _ in range(100):
        [first, second] = sorted([uniform(0, duration), uniform(0, duration)])
        assert isclose(
            audio.trim(first, second).duration(), second - first, abs_tol=1e-4
        )


def test_bandpass(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str)
    assert isinstance(s.bandpass(1, 100), Audio)


def test_bandpass(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    assert isinstance(s.bandpass(0.001, 4999), Audio)


def test_bandpass_low_error(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str)
    with pytest.raises(ValueError):
        s.bandpass(0, 100)


def test_bandpass_high_error(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    with pytest.raises(ValueError):
        s.bandpass(100, 5000)


def test_spectrum(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str)
    assert len(s.spectrum()) == 2


def test_save(silence_10s_mp3_str, saved_wav):
    Audio.from_file(silence_10s_mp3_str).save(saved_wav)
    assert saved_wav.exists()


def test_save_extension_error(silence_10s_mp3_str, saved_mp3):
    with pytest.raises(TypeError):
        Audio.from_file(silence_10s_mp3_str).save(saved_mp3)


def test_audio_constructor_should_fail_on_file(veryshort_wav_str):
    with pytest.raises(ValueError):
        Audio(veryshort_wav_str, 22050)


def test_audio_constructor_should_fail_on_non_integer_sample_rate():
    with pytest.raises(ValueError):
        Audio(np.zeros(10), "fail...")

# def test_split_and_save_dry(silence_10s_mp3_str, saved_wav):
#     Audio.from_file(silence_10s_mp3_str).split_and_save(
#         clip_length=5,
#         destination=saved_wav.parent,
#         name=saved_wav.name,
#         create_log=False,
#         final_clip=None,
#         dry=True,
#     )
    
# def test_split_and_save(silence_10s_mp3_str, saved_wav,tmp_dir):
#     df = Audio.from_file(silence_10s_mp3_str).split_and_save(
#         clip_length=5,
#         destination=saved_wav.parent,
#         name=saved_wav.name,
#         create_log=False,
#         final_clip=None,
#         dry=False,
#     )
#     for p in df.index:
#         path = tmp_dir.joinpath(Path(p))
#         path.unlink()
#     path = tmp_dir.joinpath(Path(saved_wav.name+'_clip_log.csv'))
#     path.unlink()

# def test_split_and_save_full(silence_10s_mp3_str, saved_wav):
#     Audio.from_file(silence_10s_mp3_str).split_and_save(
#         clip_length=4,
#         destination=saved_wav.parent,
#         name=saved_wav.name,
#         create_log=False,
#         final_clip="full",
#         dry=True,
#     )
    
# def test_split_and_save_short(silence_10s_mp3_str, saved_wav):
#     Audio.from_file(silence_10s_mp3_str).split_and_save(
#         clip_length=4,
#         destination=saved_wav.parent,
#         name=saved_wav.name,
#         create_log=False,
#         final_clip="short",
#         dry=True,
#     )

# def test_split_and_save_log(silence_10s_mp3_str, saved_wav):
#     Audio.from_file(silence_10s_mp3_str).split_and_save(
#         clip_length=5,
#         destination=saved_wav.parent,
#         name=saved_wav.name,
#         create_log=False,
#         final_clip=None,
#         dry=True,
#     )
    
# def test_split_and_save_short(veryshort_wav_pathlib, saved_wav):
#     with pytest.warns(UserWarning):
#         Audio.from_file(veryshort_wav_pathlib).split_and_save(
#             clip_length=5,
#             destination=saved_wav.parent,
#             name=saved_wav.name,
#             create_log=False,
#             final_clip=None,
#             dry=True,
#         )
