import pytest
from pathlib import Path
import io
import numpy as np
import random
import math
import numpy.testing as npt
import pytz
import datetime

from opensoundscape.audio import Audio, AudioOutOfBoundsError, load_channels_as_audio
from opensoundscape import audio
import opensoundscape


@pytest.fixture()
def metadata_wav_str():
    return "tests/audio/metadata.wav"


@pytest.fixture()
def new_metadata_wav_str(request):
    path = "tests/audio/new_metadata.wav"

    def fin():
        Path("tests/audio/new_metadata.wav").unlink()

    request.addfinalizer(fin)

    return path


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
def short_no_metadata_wav_str():
    return "tests/audio/short_no_metadata.wav"


@pytest.fixture()
def veryshort_wav_audio(veryshort_wav_str):
    return Audio.from_file(veryshort_wav_str)


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
def multichannel_audio():
    return audio.MultiChannelAudio(
        samples=np.array([np.ones(10), np.ones(10)]), sample_rate=10
    )


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
def saved_ogg(request, tmp_dir):
    path = Path(f"{tmp_dir}/saved.ogg")

    def fin():
        if path.exists():
            path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def stereo_wav_str():
    return "tests/audio/stereo.wav"


@pytest.fixture()
def silent_wav_str():
    return "tests/audio/silence_10s.mp3"


@pytest.fixture()
def convolved_wav_str(out_path, request):
    path = Path(f"{out_path}/convolved.wav")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def veryshort_audio(veryshort_wav_str):
    return Audio.from_file(veryshort_wav_str)


@pytest.fixture()
def LOCA_array_3_str():
    return "tests/audio/LOCA_2021_09_24_652_3.wav"


@pytest.fixture()
def LOCA_array_6_str():
    return "tests/audio/LOCA_2021_09_24_652_6.wav"


def test_init_with_list():
    a = Audio([0] * 10, sample_rate=10)
    assert len(a.samples) == 10


def test_init_with_nparray():
    a = Audio(np.zeros(10), sample_rate=10)
    assert len(a.samples) == 10


def test_load_channels_as_audio(stereo_wav_str):
    s = load_channels_as_audio(stereo_wav_str)
    assert max(s[0].samples) == 0  # channel 1 of stereo.wav is all 0
    assert max(s[1].samples) == 1  # channel 2 of stereo.wav is all 1
    assert len(s) == 2
    assert type(s[0]) == Audio

    assert s[0].metadata["channel"] == "1 of 2"
    assert s[0].metadata["channels"] == 1


def test_load_channels_as_audio_from_mono(veryshort_wav_str):
    s = load_channels_as_audio(veryshort_wav_str)
    assert len(s) == 1
    assert type(s[0]) == Audio


def test_duration(veryshort_wav_audio):
    assert math.isclose(veryshort_wav_audio.duration, 0.14208616780045352, abs_tol=1e-7)


def test_normalize(veryshort_wav_audio):
    assert math.isclose(
        max(abs(veryshort_wav_audio.normalize(peak_level=0.5).samples)),
        0.5,
        abs_tol=1e-4,
    )


def test_normalize_default(veryshort_wav_audio):
    assert math.isclose(
        max(abs(veryshort_wav_audio.normalize().samples)),
        1.0,
        abs_tol=1e-4,
    )


def test_normalize_dBFS(veryshort_wav_audio):
    assert math.isclose(
        max(abs(veryshort_wav_audio.normalize(peak_dBFS=-3).samples)),
        0.707946,
        abs_tol=1e-4,
    )


def test_apply_gain():
    a = Audio([1, -1, 0], sample_rate=10).apply_gain(dB=-20)
    assert math.isclose(a.samples.max(), 0.1, abs_tol=1e-6)
    assert math.isclose(a.samples.min(), -0.1, abs_tol=1e-6)


def test_gain_clips():
    a = Audio([0.5, -0.5, 0], sample_rate=10).apply_gain(dB=10)
    assert math.isclose(a.samples.max(), 1, abs_tol=1e-6)
    assert math.isclose(a.samples.min(), -1, abs_tol=1e-6)


def test_normalize_by_db(veryshort_wav_audio):
    assert math.isclose(
        max(abs(veryshort_wav_audio.normalize(peak_dBFS=0).samples)), 1, abs_tol=1e-4
    )


def test_normalize_doesnt_allow_both_arguments(veryshort_wav_audio):
    with pytest.raises(ValueError):
        veryshort_wav_audio.normalize(peak_dBFS=0, peak_level=0.2)


def test_load_incorrect_timestamp(onemin_wav_str):
    with pytest.raises(AssertionError):
        timestamp = "NotATimestamp"
        s = Audio.from_file(onemin_wav_str, start_timestamp=timestamp)


def test_load_timestamp_no_metadata_raises(short_no_metadata_wav_str):
    """attempts to parse metadata to get datetime, fails because file doesn't have metadata"""
    with pytest.raises(AssertionError):  # file doesn't have audiomoth metadata
        local_timestamp = datetime.datetime(2018, 4, 5, 9, 32, 0)
        local_timezone = pytz.timezone("US/Eastern")
        timestamp = local_timezone.localize(local_timestamp)
        s = Audio.from_file(short_no_metadata_wav_str, start_timestamp=timestamp)


def test_load_timestamp_after_end_of_recording(metadata_wav_str):
    with pytest.raises(AudioOutOfBoundsError):
        local_timestamp = datetime.datetime(
            2021, 4, 4, 0, 0, 0
        )  # 1 year after recording
        local_timezone = pytz.timezone("US/Eastern")
        timestamp = local_timezone.localize(local_timestamp)
        s = Audio.from_file(
            metadata_wav_str, start_timestamp=timestamp, out_of_bounds_mode="raise"
        )


def test_load_timestamp_before_recording(metadata_wav_str):
    with pytest.raises(AudioOutOfBoundsError):
        local_timestamp = datetime.datetime(
            2018, 4, 4, 0, 0, 0
        )  # 1 year before recording
        local_timezone = pytz.timezone("UTC")
        timestamp = local_timezone.localize(local_timestamp)
        s = Audio.from_file(
            metadata_wav_str, start_timestamp=timestamp, out_of_bounds_mode="raise"
        )


def test_load_timestamp_before_warnmode(metadata_wav_str):
    with pytest.warns(UserWarning):
        correct_ts = Audio.from_file(metadata_wav_str).metadata["recording_start_time"]
        local_timestamp = datetime.datetime(
            2018, 4, 4, 0, 0, 0
        )  # 1 year before recording
        local_timezone = pytz.timezone("UTC")
        timestamp = local_timezone.localize(local_timestamp)
        s = Audio.from_file(
            metadata_wav_str, start_timestamp=timestamp, out_of_bounds_mode="warn"
        )
        # Assert the start time is the correct, original timestamp and has not been changed
        assert s.metadata["recording_start_time"] == correct_ts


def test_retain_metadata_soundfile(metadata_wav_str, new_metadata_wav_str):
    a = Audio.from_file(metadata_wav_str)
    a.save(new_metadata_wav_str, metadata_format="soundfile")
    new_a = Audio.from_file(new_metadata_wav_str)

    # file size may differ slightly, other fields should be the same
    assert new_a.metadata == a.metadata


def test_save_with_empty_metadata_field(metadata_wav_str, new_metadata_wav_str):
    a = Audio.from_file(metadata_wav_str)
    a.metadata = {"date": "", "artist": "me"}

    # should write new file with ' ' as date rather than raising an error
    a.save(new_metadata_wav_str, metadata_format="soundfile")
    new_a = Audio.from_file(new_metadata_wav_str)
    assert new_a.metadata["date"] == " "
    assert new_a.metadata["artist"] == "me"


def test_save_load_opso_metadata(metadata_wav_str, new_metadata_wav_str):
    # add more tests if more versions are added

    a = Audio.from_file(metadata_wav_str)
    a.save(new_metadata_wav_str, metadata_format="opso")
    new_md = Audio.from_file(new_metadata_wav_str).metadata

    assert new_md["opensoundscape_version"] == opensoundscape.__version__
    assert new_md["opso_metadata_version"] == "v0.1"  # current default version

    # file size may differ slightly, other fields should be the same
    # all other keys/values should be equivalent:
    del new_md["opensoundscape_version"]
    del new_md["opso_metadata_version"]
    assert new_md == a.metadata


def test_update_metadata(metadata_wav_str, new_metadata_wav_str):
    a = Audio.from_file(metadata_wav_str)
    a.metadata["artist"] = "newartist"
    a.save(new_metadata_wav_str)
    assert Audio.from_file(new_metadata_wav_str).metadata["artist"] == "newartist"


def test_read_write_metadata(metadata_wav_str, new_metadata_wav_str):
    """update metadata header without reading/writing entire file"""
    # first make both files exist with same metadata
    a = Audio.from_file(metadata_wav_str)
    a.save(new_metadata_wav_str)

    # using opso metadata format
    meta = audio.parse_metadata(metadata_wav_str)
    meta["artist"] = "newartist"
    audio.write_metadata(meta, "opso", new_metadata_wav_str)
    meta2 = audio.parse_metadata(new_metadata_wav_str)
    meta2["artist"] == "newartist"
    # repeat with soundfinder metadata format
    meta = audio.parse_metadata(metadata_wav_str)
    meta["artist"] = "newartist"
    audio.write_metadata(meta, "soundfile", new_metadata_wav_str)
    meta2 = audio.parse_metadata(new_metadata_wav_str)
    meta2["artist"] == "newartist"


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
    m_dict = audio.parse_metadata(veryshort_wav_str)
    assert m_dict["samplerate"] == 44100
    a = Audio.from_file(veryshort_wav_str)
    assert a.metadata["samplerate"] == 44100


def test_load_metadata_int_offset(metadata_wav_str):
    # addresses issue #928
    Audio.from_file(metadata_wav_str, offset=np.int32(3), duration=0.1)
    Audio.from_file(metadata_wav_str, offset=np.float32(3), duration=0.1)


# currently don't know how to create a file with bad / no metadata
# def test_load_metadata_warning(path_with_no_metadata):
#     with pytest.raises(UserWarning)
#         a=Audio.from_file(path_with_no_metadata)


def test_calculate_rms(veryshort_wav_audio):
    assert math.isclose(veryshort_wav_audio.rms, 0.0871002, abs_tol=1e-7)


def test_calculate_dBFS(veryshort_wav_audio):
    assert math.isclose(veryshort_wav_audio.dBFS, -18.189316963185377, abs_tol=1e-5)


def test_property_trim_length_is_correct(silence_10s_mp3_str):
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    duration = audio.duration
    for _ in range(100):
        [first, second] = sorted(
            [random.uniform(0, duration), random.uniform(0, duration)]
        )
        assert math.isclose(
            audio.trim(first, second).duration, second - first, abs_tol=1e-4
        )


def test_trim_updates_metadata(metadata_wav_str):
    a = Audio.from_file(metadata_wav_str)
    a2 = a.trim(1, 2)
    assert a2.metadata["recording_start_time"] == a.metadata[
        "recording_start_time"
    ] + datetime.timedelta(seconds=1)
    assert a2.metadata["duration"] == 1


def test_trim_with_timestamps_metadata(metadata_wav_str):
    a = Audio.from_file(metadata_wav_str)
    start_time = a.metadata["recording_start_time"]
    a2 = a.trim_with_timestamps(
        start_time + datetime.timedelta(seconds=1),
        start_time + datetime.timedelta(seconds=2),
    )
    assert a2.metadata["recording_start_time"] == start_time + datetime.timedelta(
        seconds=1
    )
    assert a2.metadata["duration"] == 1
    assert a2.duration == 1

    # repeat with `duration` argument:
    a3 = a.trim_with_timestamps(start_time + datetime.timedelta(seconds=1), duration=1)
    assert a3.metadata["recording_start_time"] == start_time + datetime.timedelta(
        seconds=1
    )
    assert a3.metadata["duration"] == 1
    assert a3.duration == 1


def test_trim_with_timestamps_raises_error_if_tz_is_not_localized(metadata_wav_str):
    # check that it raises error if tz is not localized
    a = Audio.from_file(metadata_wav_str)
    with pytest.raises(AssertionError):
        a.trim_with_timestamps(
            datetime.datetime.now(), duration=1, out_of_bounds_mode="ignore"
        )


def test_trim_timestamps_raises_out_of_bounds_error(metadata_wav_str):
    # raises error if requested time is out of bounds and out_of_bounds_error is True
    a = Audio.from_file(metadata_wav_str)
    start_time = a.metadata["recording_start_time"]
    with pytest.raises(AudioOutOfBoundsError):
        # start too early
        a.trim_with_timestamps(
            start_time - datetime.timedelta(seconds=3),
            duration=1,
            out_of_bounds_mode="raise",
        )
    with pytest.raises(AudioOutOfBoundsError):
        # end too late
        a.trim_with_timestamps(start_time, duration=1000, out_of_bounds_mode="raise")
    # no error if out_of_bounds_error is False
    a.trim_with_timestamps(
        start_time - datetime.timedelta(seconds=3),
        duration=1000,
        out_of_bounds_mode="ignore",
    )


def test_trim_from_negative_time(silence_10s_mp3_str):
    """correct behavior is to trim from time zero"""
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    a = audio.trim(-1, 5)
    assert math.isclose(a.duration, 5, abs_tol=1e-5)

    with pytest.warns(UserWarning):
        audio.trim(-1, 5, out_of_bounds_mode="warn")

    with pytest.raises(AudioOutOfBoundsError):
        audio.trim(-1, 5, out_of_bounds_mode="raise")


def test_trim_samples(silence_10s_mp3_str):
    """correct behavior is to trim from time zero"""
    audio = Audio.from_file(silence_10s_mp3_str)
    assert len(audio.trim_samples(0, 10).samples) == 10

    assert len(audio.trim_samples(200, 210).samples) == 10

    assert len(audio.trim_samples(-10, 10).samples) == 10

    assert len(audio.trim_samples(10, 10).samples) == 0

    with pytest.raises(AssertionError):
        # cannot pass start index > end index
        audio.trim_samples(20, 10)


def test_trim_past_end_of_clip(silence_10s_mp3_str):
    """correct behavior is to trim to the end of the clip"""
    a = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    audio = a.trim(9, 11)
    assert math.isclose(audio.duration, 1, abs_tol=1e-5)

    with pytest.warns(UserWarning):
        a.trim(9, 11, out_of_bounds_mode="warn")

    with pytest.raises(AudioOutOfBoundsError):
        a.trim(9, 11, out_of_bounds_mode="raise")


def test_trim_with_datetime(silence_10s_mp3_str):
    a = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    a.metadata["recording_start_time"] = datetime.datetime(
        2022, 1, 1, 0, 0, 0, tzinfo=pytz.utc
    )
    start = datetime.datetime(2022, 1, 1, 0, 0, 1, tzinfo=pytz.utc)
    end = datetime.datetime(2022, 1, 1, 0, 0, 2, tzinfo=pytz.utc)
    audio = a.trim_with_timestamps(start, end)
    assert math.isclose(audio.duration, 1, abs_tol=1e-5)


def test_resample_veryshort_wav(veryshort_wav_str):
    """Note: default resample type changed from kaiser_Fast to soxr_hq

    this change mirrors default change in librosa, and means we need to
    require librosa>=0.10.0 or add soxr as a dependency. See issue:
    https://github.com/kitzeslab/opensoundscape/issues/674

    I've chosen to add librosa>=0.10.0 as a dependency.
    """
    audio = Audio.from_file(veryshort_wav_str)
    dur = audio.duration
    resampled_audio = audio.resample(22050)
    assert np.isclose(resampled_audio.duration, dur, 1e-6)
    assert resampled_audio.samples.shape == (3133,)
    assert resampled_audio.sample_rate == 22050


def test_spawn(veryshort_wav_audio):
    """spawn method creates copy of Audio object with any kwarg fields updated"""
    a = veryshort_wav_audio
    a2 = a._spawn()
    assert np.all(a2.samples == a.samples)
    assert a2.sample_rate == a.sample_rate

    # provide an updated value for the new object
    a3 = a._spawn(sample_rate=20)
    assert a3.sample_rate == 20

    with pytest.raises(AssertionError):
        # should not be able to pass non __slot__ kwargs
        a._spawn(b=1)


def test_methods_retain_metadata(metadata_wav_str):
    """resolved issue 679 by implementing ._spawn
    This should avoid future issues with improperly calling
    Audio.__init__ ie if the arguments to __init__ change
    """
    audio = Audio.from_file(metadata_wav_str)
    a2 = audio.resample(22050)
    assert audio.metadata == a2.metadata

    a2 = audio.apply_gain(-2)
    assert audio.metadata == a2.metadata

    a2 = audio.normalize()
    assert audio.metadata == a2.metadata


def test_resample_mp3_nonstandard_sr(silence_10s_mp3_str):
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    dur = audio.duration
    resampled_audio = audio.resample(5000)
    assert resampled_audio.duration == dur
    assert resampled_audio.sample_rate == 5000


def test_resample_classmethod_vs_instancemethod(silence_10s_mp3_str):
    a1 = Audio.from_file(silence_10s_mp3_str)
    a1 = a1.resample(2000)
    a2 = Audio.from_file(silence_10s_mp3_str, sample_rate=2000)
    npt.assert_array_almost_equal(a1.samples, a2.samples, decimal=5)


def test_extend_to_length_is_correct(silence_10s_mp3_str):
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    duration = audio.duration
    for _ in range(100):
        extend_length = random.uniform(duration, duration * 10)
        assert math.isclose(
            audio.extend_to(extend_length).duration, extend_length, abs_tol=1e-4
        )


def test_extend_to_correct_metadata(silence_10s_mp3_str):
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    a2 = audio.extend_to(12)
    # duration in metadata should be updated:
    assert math.isclose(a2.metadata["duration"], 12)
    # other metadata should be retained:
    assert a2.metadata["subtype"] == audio.metadata["subtype"]


def test_extend_to_shorter_duration(silence_10s_mp3_str):
    # extending 10s to 6s should retain 10s
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    a2 = audio.extend_to(6)
    assert math.isclose(a2.duration, 10)
    # duration in metadata should be updated:
    assert math.isclose(a2.metadata["duration"], 10)
    # other metadata should be retained:
    assert a2.metadata["subtype"] == audio.metadata["subtype"]


def test_extend_to_correct_duration_ok(silence_10s_mp3_str):
    # extending 10s to 10 shouldn't raise error (#972)
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    a2 = audio.extend_to(10)
    assert math.isclose(a2.duration, 10)
    # duration in metadata should be updated:
    assert math.isclose(a2.metadata["duration"], 10)
    # other metadata should be retained:
    assert a2.metadata["subtype"] == audio.metadata["subtype"]


def test_extend_by(silence_10s_mp3_str):
    audio = Audio.from_file(silence_10s_mp3_str, sample_rate=10000)
    a2 = audio.extend_by(1)
    assert math.isclose(a2.duration, 11)

    # duration in metadata should be updated:
    assert math.isclose(a2.metadata["duration"], 11)
    # other metadata should be retained:
    assert a2.metadata["subtype"] == audio.metadata["subtype"]

    # doesn't allow negative extend_by(duration)
    with pytest.raises(AssertionError):
        a2.extend_by(-1)


def test_bandpass(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str)
    assert isinstance(s.bandpass(1, 100, 9), Audio)


def test_lowpass(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str)
    assert isinstance(s.lowpass(100, 9), Audio)


def test_highpass(silence_10s_mp3_str):
    s = Audio.from_file(silence_10s_mp3_str)
    assert isinstance(s.highpass(100, 9), Audio)


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


def test_save_mp3(silence_10s_mp3_str, saved_mp3):
    try:
        Audio.from_file(silence_10s_mp3_str).save(saved_mp3)
        assert saved_mp3.exists()
        Audio.from_file(saved_mp3)  # make sure we can still load it as audio
    except NotImplementedError:
        # only supported by libsndfile>=1.1.0, which is not available yet
        # on ubuntu as of Dec 2022. So, we just give the user a helpful error.
        pass


def test_save_specific_subtype(silence_10s_mp3_str, saved_ogg):
    Audio.from_file(silence_10s_mp3_str, sample_rate=16000).save(
        saved_ogg, format="OGG", subtype="OPUS"
    )
    assert saved_ogg.exists()
    Audio.from_file(saved_ogg)  # make sure we can still load it as audio


def test_save_specific_compression_level(silence_10s_mp3_str, saved_mp3):
    Audio.from_file(silence_10s_mp3_str).save(saved_mp3, compression_level=0.8)
    assert saved_mp3.exists()
    Audio.from_file(saved_mp3)  # make sure we can still load it as audio


def test_save_specific_bitrate_mode(silence_10s_mp3_str, saved_mp3):
    Audio.from_file(silence_10s_mp3_str).save(
        saved_mp3, compression_level=0.8, bitrate_mode="CONSTANT"
    )
    assert saved_mp3.exists()
    Audio.from_file(saved_mp3)  # make sure we can still load it as audio


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


def test_skip_loading_metadata(metadata_wav_str):
    a = Audio.from_file(metadata_wav_str, load_metadata=False)
    assert a.metadata is None


def test_silent_classmethod():
    a = Audio.silence(10, 200)
    assert len(a.samples) == 2000
    assert max(a.samples) == 0


def test_noise_classmethod():
    for c in ["white", "blue", "violet", "brown", "pink"]:
        a = Audio.noise(1, 200, color=c)
        assert len(a.samples) == 200


def test_concat(veryshort_wav_audio):
    a = audio.concat([veryshort_wav_audio, veryshort_wav_audio])
    assert a.duration == 2 * veryshort_wav_audio.duration


def test_mix(veryshort_wav_audio):
    m = audio.mix([veryshort_wav_audio, veryshort_wav_audio], gain=0)
    assert math.isclose(
        max(veryshort_wav_audio.samples) * 2, max(m.samples), abs_tol=1e-6
    )


def test_mix_duration(veryshort_wav_audio):
    m = audio.mix([veryshort_wav_audio, veryshort_wav_audio], duration=1)
    assert math.isclose(m.duration, 1, abs_tol=1e-6)


def test_mix_duration_extends(veryshort_wav_audio):
    m = audio.mix([veryshort_wav_audio, veryshort_wav_audio], duration=1)
    assert math.isclose(m.duration, 1, abs_tol=1e-3)


def test_mix_duration_truncates(veryshort_wav_audio):
    a = Audio.silence(10, veryshort_wav_audio.sample_rate)
    m = audio.mix([a, veryshort_wav_audio], duration=1)
    assert math.isclose(m.duration, 1, abs_tol=1e-3)


def test_mix_offsets(veryshort_wav_audio):
    m = audio.mix([veryshort_wav_audio, veryshort_wav_audio], offsets=[0, 1])
    # expected behavior: length will be offset + audio length
    assert math.isclose(m.duration, 1 + veryshort_wav_audio.duration, abs_tol=1e-3)


def test_mix_clip(veryshort_wav_audio):
    # should never have values outside of [-1,1]
    m = audio.mix([veryshort_wav_audio, veryshort_wav_audio], gain=100)
    assert max(abs(m.samples)) <= 1


def test_loop(veryshort_wav_audio):
    veryshort_wav_audio.metadata = {"duration": veryshort_wav_audio.duration}
    a2 = veryshort_wav_audio.loop(n=2)
    assert math.isclose(a2.duration, veryshort_wav_audio.duration * 2, abs_tol=1e-5)
    assert math.isclose(a2.duration, a2.metadata["duration"])

    a3 = veryshort_wav_audio.loop(length=1)
    assert math.isclose(a3.duration, 1.0, abs_tol=1e-5)
    assert math.isclose(a3.metadata["duration"], 1.0, abs_tol=1e-5)


def test_extend_to_with_short(veryshort_wav_audio):
    a = veryshort_wav_audio.extend_to(duration=1)
    assert math.isclose(a.duration, 1.0, abs_tol=1e-5)
    assert math.isclose(a.metadata["duration"], 1.0, abs_tol=1e-5)
    # added samples should be zero
    assert math.isclose(0.0, np.max(a.samples[-100:]), abs_tol=1e-7)


def test_bandpass_filter(veryshort_audio):
    bandpassed = audio.bandpass_filter(
        veryshort_audio.samples, 1000, 2000, veryshort_audio.sample_rate
    )
    assert len(bandpassed) == len(veryshort_audio.samples)


def test_reduce_noise():
    from opensoundscape.utils import set_seed

    set_seed(0)
    noise = Audio.noise(1, sample_rate=8000, color="white")
    reduced = noise.reduce_noise()
    assert reduced.rms < noise.rms


def test_clipping_detector(veryshort_audio):
    assert audio.clipping_detector(veryshort_audio.samples) > -1


def test_estimate_delay(veryshort_audio):
    start_t = 0.03
    end_t = 0.12
    signal = veryshort_audio.trim(start_time=start_t, end_time=end_t)
    delay = 0.02
    max_delay = 0.03
    ref_sig = veryshort_audio.trim(start_time=start_t + delay, end_time=end_t + delay)

    # phat filter will fail on the below
    # maybe because the signal is too short
    assert math.isclose(
        audio.estimate_delay(signal, ref_sig, max_delay=max_delay, cc_filter="cc_norm"),
        delay,
        abs_tol=1e-4,
    )


def test_estimate_delay_real_audio(LOCA_array_3_str, LOCA_array_6_str):
    max_delay = 0.1
    start_time = 0.2
    duration = 0.4
    audio_3_reference = Audio.from_file(
        LOCA_array_3_str,
        offset=start_time,
        duration=duration,
    )
    audio_6 = Audio.from_file(
        LOCA_array_6_str,
        offset=start_time,
        duration=duration,
    )
    real_measured_delay = 0.03
    delay = audio.estimate_delay(audio_6, audio_3_reference, max_delay=max_delay)
    assert abs(delay) < max_delay
    assert np.isclose(delay, real_measured_delay, atol=5e-3)


def test_estimate_delay_with_bandpass(veryshort_audio):
    start_t = 0.03
    end_t = 0.1
    signal = veryshort_audio.trim(start_time=start_t, end_time=end_t)
    delay = 0.01
    max_delay = 0.02
    ref_sig = veryshort_audio.trim(start_time=start_t + delay, end_time=end_t + delay)

    dly = audio.estimate_delay(
        signal,
        ref_sig,
        max_delay=max_delay,
        bandpass_range=[100, 10000],
        bandpass_order=5,
    )
    assert math.isclose(dly, delay, abs_tol=1e-4)


def test_estimate_delay_return_cc_max(veryshort_audio):
    max_delay = 0.05
    delay, ccmax = audio.estimate_delay(
        veryshort_audio,
        veryshort_audio,
        max_delay=max_delay,
        cc_filter="cc",
        return_cc_max=True,
    )

    # will use only the central part of the signal
    section_used = veryshort_audio.trim(
        start_time=max_delay, end_time=veryshort_audio.duration - max_delay
    )
    assert math.isclose(
        ccmax, sum(section_used.samples * section_used.samples), abs_tol=1e-5
    )
    assert math.isclose(delay, 0, abs_tol=1e-6)


def test_from_url_multichannel_to_mono():
    """note: test will fail if the file is removed from xeno-canto
    or is inaccessible at this url

    downloads a 2-channel audio file and sums to 1, ensuring resolution of #837
    """
    Audio.from_url("https://tinyurl.com/birds60s")


def test_multichannelaudio_init():
    a = audio.MultiChannelAudio(np.zeros((2, 10)), sample_rate=10)
    assert a.samples.shape == (2, 10)
    assert a.sample_rate == 10
    assert a.n_channels == 2
    assert a.duration == 1


def test_multichannelaudio_from_file(veryshort_wav_str, stereo_wav_str):
    a = audio.MultiChannelAudio.from_file(veryshort_wav_str)
    assert a.samples.shape == (1, 6266)
    assert a.sample_rate == 44100
    assert a.n_channels == 1
    assert np.isclose(a.duration, 0.14208616780045352, atol=1e-5)

    a = audio.MultiChannelAudio.from_file(stereo_wav_str)
    assert max(a.samples[0, :]) == 0  # channel 1 of stereo.wav is all 0
    assert max(a.samples[1, :]) == 1  # channel 2 of stereo.wav is all 1
    assert a.n_channels == 2


def test_multichannelaudio_trim(multichannel_audio):
    a2 = multichannel_audio.trim(0, 0.5)
    assert np.isclose(a2.duration, 0.5, atol=1e-5)


def test_multichannelaudio_apply_gain(multichannel_audio):
    a2 = multichannel_audio.apply_gain(dB=-20)
    assert math.isclose(a2.samples.max(), 0.1, abs_tol=1e-6)


def test_multichannelaudio_apply_channel_gain(multichannel_audio):
    a2 = multichannel_audio.apply_channel_gain(dB=[-20, 20], clip_range=[-5, 5])
    assert math.isclose(a2.samples[0, :].max(), 0.1, abs_tol=1e-6)
    assert math.isclose(a2.samples[1, :].max(), 5, abs_tol=1e-6)


def test_mix_multichannel(multichannel_audio):
    m = audio.mix([multichannel_audio, multichannel_audio], gain=[-20, -20])
    assert math.isclose(max(m.samples[0, :]), 0.2, abs_tol=1e-6)


def test_mix_multichannel_and_mono(multichannel_audio, veryshort_wav_audio):
    m = audio.mix([multichannel_audio, veryshort_wav_audio])
    assert m.n_channels == 2


def test_multichannel_extend_to(multichannel_audio):
    a = multichannel_audio.extend_to(duration=2)
    assert math.isclose(a.duration, 2.0, abs_tol=1e-5)
    # added samples should be zero
    assert math.isclose(0.0, np.max(a.samples[0, -10:]), abs_tol=1e-7)


def test_multichannel_extend_by(multichannel_audio):
    a = multichannel_audio.extend_by(1)
    assert math.isclose(a.duration, 2.0, abs_tol=1e-5)
    # added samples should be zero
    assert math.isclose(0.0, np.max(a.samples[0, -10:]), abs_tol=1e-7)


def test_save_load_multichannel(multichannel_audio, saved_wav):
    multichannel_audio.metadata = {"artist": "test"}
    multichannel_audio.save(saved_wav)
    assert saved_wav.exists()
    a2 = audio.MultiChannelAudio.from_file(saved_wav)
    # the tolerance is pretty bad for some reason, for saving and reloading the same sample values
    # TODO why so different? fails if atol=1e-5 or smaller
    assert np.allclose(a2.samples, multichannel_audio.samples, atol=1e-4)
    assert a2.sample_rate == multichannel_audio.sample_rate
    assert a2.n_channels == multichannel_audio.n_channels
    assert a2.duration == multichannel_audio.duration
    assert a2.metadata["artist"] == "test"


def test_multichannelaudio_silence():
    a = audio.MultiChannelAudio.silence(duration=2, channels=3, sample_rate=10)
    assert a.samples.shape == (3, 20)
    assert a.samples.max() == 0


def test_multichannelaudio_noise():
    a = audio.MultiChannelAudio.noise(duration=2, channels=3, sample_rate=10)
    assert a.samples.shape == (3, 20)
    assert type(a) == audio.MultiChannelAudio


def test_multichannelaudio_concat(multichannel_audio):
    a = audio.concat([multichannel_audio, multichannel_audio])
    assert a.duration == 2 * multichannel_audio.duration
    assert a.n_channels == multichannel_audio.n_channels


def test_multichannel_from_audio_list(veryshort_audio):
    a = audio.MultiChannelAudio.from_audio_list([veryshort_audio, veryshort_audio])
    assert a.n_channels == 2
