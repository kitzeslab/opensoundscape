#!/usr/bin/env python3
from opensoundscape.audio import Audio, OpsoLoadAudioInputTooLong, split_and_save
import pytest
from pathlib import Path
import io
import numpy as np
from random import uniform
from math import isclose


@pytest.fixture()
def veryshort_wav_str():
    return "tests/veryshort.wav"


@pytest.fixture()
def silence_10s_mp3_str():
    return "tests/silence_10s.mp3"


@pytest.fixture()
def not_a_file_str():
    return "tests/not_a_file.wav"


@pytest.fixture()
def out_path():
    return "tests/audio_out"


@pytest.fixture()
def veryshort_wav_pathlib(veryshort_wav_str):
    return Path(veryshort_wav_str)


@pytest.fixture()
def veryshort_wav_bytesio(veryshort_wav_str):
    with open(veryshort_wav_str, "rb") as f:
        return io.BytesIO(f.read())


@pytest.fixture()
def silence_10s_mp3_pathlib(silence_10s_mp3_str):
    return Path(silence_10s_mp3_str)


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


def test_extend_length_is_correct(silence_10s_mp3_str):
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    duration = audio.duration()
    for _ in range(100):
        extend_length = uniform(duration, duration * 10)
        assert isclose(
            audio.extend(extend_length).duration(), extend_length, abs_tol=1e-4
        )


def test_bandpass(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str)
    assert isinstance(s.bandpass(1, 100, 9), Audio)


def test_bandpass_sample_rate_10000(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    assert isinstance(s.bandpass(0.001, 4999, 9), Audio)


def test_bandpass_low_error(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str)
    with pytest.raises(ValueError):
        s.bandpass(0, 100, 9)


def test_bandpass_high_error(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    with pytest.raises(ValueError):
        s.bandpass(100, 5000, 9)


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


def test_split_and_save_default(silence_10s_mp3_pathlib):
    clip_df = split_and_save(
        Audio.from_file(silence_10s_mp3_pathlib),
        "unnecessary",
        "unnecessary",
        5,
        dry_run=True,
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["begin_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["begin_time"] == 4.0
    assert clip_df.iloc[1]["end_time"] == 9.0
    assert clip_df.iloc[1]["clip_duration"] == 5.0


def test_split_and_save_default_extend(silence_10s_mp3_pathlib):
    clip_df = split_and_save(
        Audio.from_file(silence_10s_mp3_pathlib),
        "unnecessary",
        "unnecessary",
        5,
        final_clip="extend",
        dry_run=True,
    )
    assert clip_df.shape[0] == 3
    assert clip_df.iloc[0]["begin_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["begin_time"] == 4.0
    assert clip_df.iloc[1]["end_time"] == 9.0
    assert clip_df.iloc[2]["begin_time"] == 8.0
    assert clip_df.iloc[2]["end_time"] == 10.0
    assert clip_df.iloc[2]["clip_duration"] == 5.0


def test_non_integer_split_and_save_default(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib).trim(0, 8.2)
    clip_df = split_and_save(audio, "unnecessary", "unnecessary", 5, dry_run=True)
    assert clip_df.shape[0] == 1
    assert clip_df.iloc[0]["begin_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[0]["clip_duration"] == 5.0


def test_non_integer_split_and_save_remainder(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib).trim(0, 8.2)
    clip_df = split_and_save(
        audio, "unnecessary", "unnecessary", 5, dry_run=True, final_clip="remainder"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["begin_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["begin_time"] == 4.0
    assert clip_df.iloc[1]["end_time"] == 8.2
    assert clip_df.iloc[1]["clip_duration"] == 4.2
