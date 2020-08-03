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
from numpy.testing import assert_array_equal


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
        print(audio.extend(extend_length).duration())
        print(extend_length)
        assert isclose(
            audio.extend(extend_length).duration(), extend_length, abs_tol=1e-4
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


def test_split_and_save_dry(silence_10s_mp3_pathlib, tmp_dir):
    Audio.from_file(silence_10s_mp3_pathlib).split_and_save(
        clip_length=5,
        destination=tmp_dir,
        name=silence_10s_mp3_pathlib.stem,
        create_log=False,
        final_clip=None,
        dry=True,
    )


def test_split_and_save(silence_10s_mp3_pathlib, tmp_dir):
    clip_length = 5
    og = Audio.from_file(silence_10s_mp3_pathlib)
    df = og.split_and_save(
        clip_length=clip_length,
        destination=tmp_dir,
        name=silence_10s_mp3_pathlib.stem,
        create_log=False,
        final_clip=None,
        dry=False,
    )
    assert df.shape[0] == 2
    for idx, p in enumerate(df.index):
        path = tmp_dir.joinpath(Path(p))
        assert_array_equal(
            og.trim(idx * clip_length, (idx + 1) * clip_length).samples,
            Audio.from_file(path).samples,
        )
        path.unlink()


def test_split_and_save_full(silence_10s_mp3_pathlib, tmp_dir):
    clip_length = 4
    og = Audio.from_file(silence_10s_mp3_pathlib)
    df = og.split_and_save(
        clip_length=clip_length,
        destination=tmp_dir,
        name=silence_10s_mp3_pathlib.stem,
        create_log=False,
        final_clip="full",
        dry=False,
    )
    assert df.shape[0] == 3
    assert_array_equal(
        og.trim(10 - clip_length, 10).samples,
        Audio.from_file(tmp_dir.joinpath(df.index[2])).samples,
    )
    for p in df.index:
        path = tmp_dir.joinpath(Path(p))
        path.unlink()


def test_split_and_save_short(silence_10s_mp3_pathlib, tmp_dir):
    clip_length = 4
    og = Audio.from_file(silence_10s_mp3_pathlib)
    df = og.split_and_save(
        clip_length=clip_length,
        destination=tmp_dir,
        name=silence_10s_mp3_pathlib.stem,
        create_log=False,
        final_clip="short",
        dry=False,
    )
    print(list(tmp_dir.glob("*")))
    assert df.shape[0] == 3
    assert_array_equal(
        og.trim(clip_length * (df.shape[0] - 1), 10).samples,
        Audio.from_file(tmp_dir.joinpath(df.index[2])).samples,
    )
    for p in df.index:
        path = tmp_dir.joinpath(Path(p))
        print(path)
        path.unlink()


def test_split_and_save_log(silence_10s_mp3_pathlib, tmp_dir):
    Audio.from_file(silence_10s_mp3_pathlib).split_and_save(
        clip_length=5,
        destination=tmp_dir,
        name=silence_10s_mp3_pathlib.stem,
        create_log=True,
        final_clip=None,
        dry=True,
    )

    path = tmp_dir.joinpath(Path(silence_10s_mp3_pathlib.stem + "_clip_log.csv"))
    path.unlink()


def test_split_and_save_too_short(veryshort_wav_pathlib, tmp_dir):
    with pytest.warns(UserWarning):
        Audio.from_file(veryshort_wav_pathlib).split_and_save(
            clip_length=5,
            destination=tmp_dir,
            name=veryshort_wav_pathlib.stem,
            create_log=False,
            final_clip=None,
            dry=True,
        )
