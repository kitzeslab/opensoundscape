#!/usr/bin/env python3
from opensoundscape.audio import Audio, AudioOutOfBoundsError, load_channels_as_audio
import pytest
from pathlib import Path
import io
import numpy as np
from random import uniform
from math import isclose
import numpy.testing as npt
import pytz
from datetime import datetime


@pytest.fixture()
def metadata_wav_str():
    return "tests/audio/metadata.wav"


@pytest.fixture()
def new_metadata_wav_str():
    return "tests/audio/new_metadata.wav"


@pytest.fixture()
def onemin_wav_str():
    return "tests/audio/1min.wav"


@pytest.fixture()
def empty_wav_str():
    return "tests/audio/empty_2c.wav"


@pytest.fixture()
def veryshort_wav_str():
    return "tests/audio/veryshort.wav"


@pytest.fixture()
def silence_10s_mp3_str():
    return "tests/audio/silence_10s.mp3"


@pytest.fixture()
def not_a_file_str():
    return "tests/audio/not_a_file.wav"


@pytest.fixture()
def out_path():
    return "tests/audio/audio_out"


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


@pytest.fixture()
def stereo_wav_str():
    return "tests/audio/stereo.wav"


def test_load_channels_as_audio(stereo_wav_str):
    s = load_channels_as_audio(stereo_wav_str)
    assert max(s[0].samples) == 0  # channel 1 of stereo.wav is all 0
    assert max(s[1].samples) == 1  # channel 2 of stereo.wav is all 1


def test_load_incorrect_timestamp(onemin_wav_str):
    with pytest.raises(AssertionError):
        timestamp = "NotATimestamp"
        s = Audio.from_file(onemin_wav_str, start_timestamp=timestamp)


def test_load_timestamp_notanaudiomothrecording(veryshort_wav_str):
    with pytest.raises(AssertionError):  # file doesn't have audiomoth metadata
        local_timestamp = datetime(2018, 4, 5, 9, 32, 0)
        local_timezone = pytz.timezone("US/Eastern")
        timestamp = local_timezone.localize(local_timestamp)
        s = Audio.from_file(veryshort_wav_str, start_timestamp=timestamp)


def test_load_timestamp_after_end_of_recording(metadata_wav_str):
    with pytest.raises(AudioOutOfBoundsError):
        local_timestamp = datetime(2021, 4, 4, 0, 0, 0)  # 1 year after recording
        local_timezone = pytz.timezone("US/Eastern")
        timestamp = local_timezone.localize(local_timestamp)
        s = Audio.from_file(
            metadata_wav_str, start_timestamp=timestamp, out_of_bounds_mode="raise"
        )


def test_load_timestamp_before_recording(metadata_wav_str):
    with pytest.raises(AudioOutOfBoundsError):
        local_timestamp = datetime(2018, 4, 4, 0, 0, 0)  # 1 year before recording
        local_timezone = pytz.timezone("UTC")
        timestamp = local_timezone.localize(local_timestamp)
        s = Audio.from_file(
            metadata_wav_str, start_timestamp=timestamp, out_of_bounds_mode="raise"
        )


def test_load_timestamp_before_warnmode(metadata_wav_str):
    with pytest.warns(UserWarning):
        correct_ts = Audio.from_file(metadata_wav_str).metadata["recording_start_time"]
        local_timestamp = datetime(2018, 4, 4, 0, 0, 0)  # 1 year before recording
        local_timezone = pytz.timezone("UTC")
        timestamp = local_timezone.localize(local_timestamp)
        s = Audio.from_file(
            metadata_wav_str, start_timestamp=timestamp, out_of_bounds_mode="warn"
        )
        # Assert the start time is the correct, original timestamp and has not been changed
        assert s.metadata["recording_start_time"] == correct_ts


def test_retain_metadata(metadata_wav_str, new_metadata_wav_str):
    a = Audio.from_file(metadata_wav_str)
    a.save(new_metadata_wav_str)
    new_a = Audio.from_file(new_metadata_wav_str)

    # file size may differ slightly, other fields should be the same
    new_a.metadata["filesize"] = None
    a.metadata["filesize"] = None
    assert new_a.metadata == a.metadata


def test_update_metadata(metadata_wav_str, new_metadata_wav_str):
    a = Audio.from_file(metadata_wav_str)
    a.metadata["artist"] = "newartist"
    a.save(new_metadata_wav_str)
    assert Audio.from_file(new_metadata_wav_str).metadata["artist"] == "newartist"


def test_load_empty_wav(empty_wav_str):
    with pytest.raises(AudioOutOfBoundsError):
        s = Audio.from_file(empty_wav_str, out_of_bounds_mode="raise")


def test_load_duration_too_long(veryshort_wav_str):
    with pytest.raises(AudioOutOfBoundsError):
        s = Audio.from_file(veryshort_wav_str, duration=5, out_of_bounds_mode="raise")


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


def test_load_not_a_file_asserts_not_a_file(not_a_file_str):
    with pytest.raises(FileNotFoundError):
        Audio.from_file(not_a_file_str)


def test_load_metadata(veryshort_wav_str):
    a = Audio.from_file(veryshort_wav_str)
    assert a.metadata["samplerate"] == 44100


# currently don't know how to create a file with bad / no metadata
# def test_load_metadata_warning(path_with_no_metadata):
#     with pytest.raises(UserWarning)
#         a=Audio.from_file(path_with_no_metadata)


def test_property_trim_length_is_correct(silence_10s_mp3_str):
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    duration = audio.duration()
    for _ in range(100):
        [first, second] = sorted([uniform(0, duration), uniform(0, duration)])
        assert isclose(
            audio.trim(first, second).duration(), second - first, abs_tol=1e-4
        )


def test_trim_from_negative_time(silence_10s_mp3_str):
    """correct behavior is to trim from time zero"""
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000).trim(-1, 5)
    assert isclose(audio.duration(), 5, abs_tol=1e-5)


def test_trim_past_end_of_clip(silence_10s_mp3_str):
    """correct behavior is to trim to the end of the clip"""
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000).trim(9, 11)
    assert isclose(audio.duration(), 1, abs_tol=1e-5)


def test_resample_veryshort_wav(veryshort_wav_str):
    audio = Audio.from_file(veryshort_wav_str)
    dur = audio.duration()
    resampled_audio = audio.resample(22050)
    assert resampled_audio.duration() == dur
    assert resampled_audio.samples.shape == (3133,)


def test_resample_mp3_nonstandard_sr(silence_10s_mp3_str):
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    dur = audio.duration()
    resampled_audio = audio.resample(5000)
    assert resampled_audio.duration() == dur
    assert resampled_audio.sample_rate == 5000


def test_resample_classmethod_vs_instancemethod(silence_10s_mp3_str):
    a1 = Audio.from_file(silence_10s_mp3_str)
    a1 = a1.resample(2000)
    a2 = Audio.from_file(silence_10s_mp3_str, sample_rate=2000)
    npt.assert_array_almost_equal(a1.samples, a2.samples, decimal=5)


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
    audio = Audio.from_file(silence_10s_mp3_pathlib)
    clip_df = audio.split_and_save("unnecessary", "unnecessary", 5.0, dry_run=True)
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 5.0
    assert clip_df.iloc[1]["end_time"] == 10.0


def test_split_and_save_default_overlay(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib)
    clip_df = audio.split_and_save("unnecessary", "unnecessary", 5.0, 1.0, dry_run=True)
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 4.0
    assert clip_df.iloc[1]["end_time"] == 9.0


def test_split_and_save_default_full(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib)
    clip_df = audio.split_and_save(
        "unnecessary", "unnecessary", 5.0, 1.0, final_clip="full", dry_run=True
    )
    assert clip_df.shape[0] == 3
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 4.0
    assert clip_df.iloc[1]["end_time"] == 9.0
    assert clip_df.iloc[2]["start_time"] == 5.0
    assert clip_df.iloc[2]["end_time"] == 10.0


def test_split_and_save_default_extend(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib)
    clip_df = audio.split_and_save(
        "unnecessary", "unnecessary", 5.0, 1.0, final_clip="extend", dry_run=True
    )
    assert clip_df.shape[0] == 3
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 4.0
    assert clip_df.iloc[1]["end_time"] == 9.0
    assert clip_df.iloc[2]["start_time"] == 8.0
    assert clip_df.iloc[2]["end_time"] == 13.0


def test_non_integer_source_split_and_save_default(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib).trim(0, 8.2)
    clip_df = audio.split_and_save("unnecessary", "unnecessary", 5, dry_run=True)
    assert clip_df.shape[0] == 1
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0


def test_non_integer_source_split_and_save_remainder(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib).trim(0, 8.2)
    clip_df = audio.split_and_save(
        "unnecessary", "unnecessary", 5, dry_run=True, final_clip="remainder"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 5.0
    assert abs(clip_df.iloc[1]["end_time"] - 8.2) < 0.1


def test_non_integer_source_split_and_save_full(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib).trim(0, 8.2)
    clip_df = audio.split_and_save(
        "unnecessary", "unnecessary", 5, dry_run=True, final_clip="full"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert abs(clip_df.iloc[1]["start_time"] - 3.2) < 0.1
    assert abs(clip_df.iloc[1]["end_time"] - 8.2) < 0.1


def test_non_integer_source_split_and_save_extend(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib).trim(0, 8.2)
    clip_df = audio.split_and_save(
        "unnecessary", "unnecessary", 5, dry_run=True, final_clip="extend"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 5.0
    assert (clip_df.iloc[1]["end_time"] - 10.0) < 0.1


def test_non_integer_cliplen_split_and_save(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib)
    clip_df = audio.split_and_save("unnecessary", "unnecessary", 4.5, dry_run=True)
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 4.5
    assert clip_df.iloc[1]["start_time"] == 4.5
    assert clip_df.iloc[1]["end_time"] == 9.0


def test_non_integer_overlaplen_split_and_save(silence_10s_mp3_pathlib):
    audio = Audio.from_file(silence_10s_mp3_pathlib)
    clip_df = audio.split_and_save("unnecessary", "unnecessary", 5.0, 0.5, dry_run=True)
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 4.5
    assert clip_df.iloc[1]["end_time"] == 9.5
