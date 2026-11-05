import numpy as np
import pytest
import pandas as pd
import pytz
import datetime
import torch
import random
import shutil
from pathlib import Path
from opensoundscape.ml import cnn, cnn_architectures

from opensoundscape import utils, audio


@pytest.fixture()
def silence_10s_mp3_str():
    return "tests/audio/silence_10s.mp3"


@pytest.fixture()
def metadata_wav_str():
    return "tests/audio/metadata.wav"


def test_isnan():
    assert not utils.isNan(0) and utils.isNan(np.nan)


def test_sigmoid():
    utils.sigmoid(-1)


def test_binarize():
    assert np.sum(utils.binarize([-1, 1], 0)) == 1


def test_binarize_2d():
    assert np.sum(utils.binarize([[0, 0.2], [5, 0.6]], 0.5)) == 2


def test_binarize_shape_error():
    with pytest.raises(ValueError):
        utils.binarize([[[0, 0.2], [5, 0.6]]], 0.5)


def test_rescale_features():
    x = utils.rescale_features([1, 2, 3], [1])
    assert x[0][0] == 1


def test_min_max_scale():
    scaled = utils.min_max_scale([-5, 10.2], (0, 1))
    assert round(min(scaled)) == 0 and round(max(scaled)) == 1


def test_jitter():
    utils.jitter([1, 2, 3], 1, distribution="gaussian")
    utils.jitter([1, 2, 3], 1, distribution="uniform")


def test_jitter_nonexistant_raises_value_error():
    with pytest.raises(ValueError):
        utils.jitter([1, 2, 3], 1, distribution="nonexistant")


def test_generate_clip_times_df_default():
    """many corner cases / alternatives are tested for audio.split()"""
    clip_df = utils.generate_clip_times_df(full_duration=10, clip_duration=5.0)
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 5.0
    assert clip_df.iloc[1]["end_time"] == 10.0


def test_floating_point_error():
    """accessing un-rounded float index causes KeyError"""
    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=0.2, rounding_precision=None
    )
    clip_df = clip_df.set_index("start_time")
    with pytest.raises(KeyError):
        clip_df.loc[0.6]


def test_rounding_avoids_fp_error():
    """default behavior rounds times to avoid key error"""
    clip_df = utils.generate_clip_times_df(
        full_duration=10,
        clip_duration=0.2,  # rounding_precision=10 default
    )
    clip_df = clip_df.set_index("start_time")
    clip_df.loc[0.6]


def test_generate_clip_times_df_extend():
    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=6.0, final_clip="extend"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 6.0
    assert clip_df.iloc[1]["start_time"] == 6.0
    assert clip_df.iloc[1]["end_time"] == 12.0


def test_generate_clip_times_df_remainder():
    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=6.0, final_clip="remainder"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 6.0
    assert clip_df.iloc[1]["start_time"] == 6.0
    assert clip_df.iloc[1]["end_time"] == 10.0


def test_generate_clip_times_df_full():
    clip_df = utils.generate_clip_times_df(
        full_duration=11, clip_duration=6.0, final_clip="full"
    )
    assert clip_df.shape[0] == 2
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 6.0
    assert clip_df.iloc[1]["start_time"] == 5.0
    assert clip_df.iloc[1]["end_time"] == 11.0


def test_generate_clip_times_df_overlap():
    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=5, clip_overlap=2.5
    )
    assert clip_df.shape[0] == 4
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 2.5
    assert clip_df.iloc[1]["end_time"] == 7.5

    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=5, overlap_fraction=0.5, final_clip=None
    )
    assert clip_df.shape[0] == 3
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 2.5
    assert clip_df.iloc[1]["end_time"] == 7.5

    clip_df = utils.generate_clip_times_df(
        full_duration=10, clip_duration=5, clip_step=2.5
    )
    assert clip_df.shape[0] == 4  # final clip default mode now "extend"
    assert clip_df.iloc[0]["start_time"] == 0.0
    assert clip_df.iloc[0]["end_time"] == 5.0
    assert clip_df.iloc[1]["start_time"] == 2.5
    assert clip_df.iloc[-1]["end_time"] == 12.5


def test_generate_clip_times_df_overlap_raises_overspecified():
    with pytest.raises(ValueError):
        utils.generate_clip_times_df(
            full_duration=10,
            clip_duration=5,
            clip_overlap=2.5,
            overlap_fraction=0.5,
        )
    with pytest.raises(ValueError):
        utils.generate_clip_times_df(
            full_duration=10,
            clip_duration=5,
            clip_overlap=2.5,
            clip_step=0.5,
        )
    with pytest.raises(ValueError):
        utils.generate_clip_times_df(
            full_duration=10,
            clip_duration=5,
            overlap_fraction=0.5,
            clip_step=0.5,
        )


def test_make_clip_df(silence_10s_mp3_str):
    """many corner cases / alternatives are tested for audio.split()

    by default, notafile.wav makes 1 row with nan as start_time and end_time
    (controlled by raise_exceptions argument)
    """
    clip_df, invalid_samples = utils.make_clip_df(
        files=[silence_10s_mp3_str, silence_10s_mp3_str, "notafile.wav"],
        clip_duration=5.0,
        return_invalid_samples=True,
    )
    assert len(clip_df) == 5
    assert len(invalid_samples) == 1


def test_make_clip_df_audio_root():
    """many corner cases / alternatives are tested for audio.split()

    by default, notafile.wav makes 1 row with nan as start_time and end_time
    (controlled by raise_exceptions argument)
    """
    clip_df, invalid_samples = utils.make_clip_df(
        files=["silence_10s.mp3"],
        clip_duration=5.0,
        return_invalid_samples=True,
        audio_root="tests/audio/",
    )
    assert len(clip_df) == 2


def test_make_clip_df_raise(silence_10s_mp3_str):
    """many corner cases / alternatives are tested for audio.split()"""
    with pytest.raises(utils.GetDurationError):
        clip_df, invalid_samples = utils.make_clip_df(
            files=[silence_10s_mp3_str, silence_10s_mp3_str, metadata_wav_str],
            clip_duration=5.0,
            return_invalid_samples=True,
            raise_exceptions=True,
        )


def test_make_clip_df_from_label_df(silence_10s_mp3_str, metadata_wav_str):
    label_df = pd.DataFrame(
        {"a": [0, 1, 2]},
        index=[silence_10s_mp3_str, silence_10s_mp3_str, metadata_wav_str],
    )
    clip_df = utils.make_clip_df(label_df, clip_duration=5.0, final_clip=None)

    # should copy labels for each file to all clips of that file
    # duplicate file should have labels from _first_ occurrence in label_df
    assert np.array_equal(clip_df["a"].values, [0, 0, 0, 0, 2, 2])


# The @pytest.mark.parametrize decorator loops trough each value in list when running pytest.
# If you add --verbose, it also prints if it passed for each value in the list for each function
# that takes it as input.

# For all utils.set_seed() tests, assert that results are determistic for the the same seed AND
# for different seeds, in a tensor/array at least one element is different.


@pytest.mark.parametrize("input", [1, 11, 13, 42, 59, 666, 1234])
def test_torch_rand(input):
    utils.set_seed(input)
    tr1 = torch.rand(100)

    utils.set_seed(input)
    tr2 = torch.rand(100)

    utils.set_seed(input + 1)
    tr3 = torch.rand(100)

    assert all(tr1 == tr2) & any(tr1 != tr3)


@pytest.mark.parametrize("input", [1, 11, 13, 42, 59, 666, 1234])
def test_numpy_random_rand(input):
    utils.set_seed(input)
    nr1 = np.random.rand(100)

    utils.set_seed(input)
    nr2 = np.random.rand(100)

    utils.set_seed(input + 1)
    nr3 = np.random.rand(100)

    assert all(nr1 == nr2) & any(nr1 != nr3)


@pytest.mark.parametrize("input", [1, 11, 13, 42, 59, 666, 1234])
def test_radom_sample(input):
    list1000 = list(range(1, 1000))

    utils.set_seed(input)
    rs1 = random.sample(list1000, 100)

    utils.set_seed(input)
    rs2 = random.sample(list1000, 100)

    utils.set_seed(input + 1)
    rs3 = random.sample(list1000, 100)

    assert (rs1 == rs2) & (rs1 != rs3)


@pytest.mark.parametrize("input", [1, 11, 13, 42, 59, 666, 1234])
def test_set_seed_results_in_deterministic_weights_cnn_init(input):
    """initializing a CNN with random weights should be deterministic after running utils.set_seed()"""
    utils.set_seed(input)
    model_resnet1 = cnn_architectures.resnet18(num_classes=10, weights=None)
    lw1 = model_resnet1.layer1[0].conv1.weight

    utils.set_seed(input)
    model_resnet2 = cnn_architectures.resnet18(num_classes=10, weights=None)
    lw2 = model_resnet2.layer1[0].conv1.weight

    utils.set_seed(input + 1)
    model_resnet3 = cnn_architectures.resnet18(num_classes=10, weights=None)
    lw3 = model_resnet3.layer1[0].conv1.weight

    assert torch.all(lw1 == lw2) & torch.any(lw1 != lw3)


def test_cast_np_to_native():
    """test that np int and float are cast to native, other types unaffected"""
    assert isinstance(utils.cast_np_to_native(np.int32(1)), int)
    assert isinstance(utils.cast_np_to_native(np.float32(1.0)), float)
    # should not affect other dtypes
    assert isinstance(utils.cast_np_to_native(True), bool)
    assert isinstance(utils.cast_np_to_native("a"), str)


@pytest.mark.parametrize("overnight", [False, True])
def test_select_recording_start_times(monkeypatch, overnight):
    starts = {
        "a": datetime.datetime(2024, 1, 2, 1, tzinfo=datetime.timezone.utc),
        "b": datetime.datetime(2024, 1, 2, 12, tzinfo=datetime.timezone.utc),
    }
    monkeypatch.setattr(
        audio, "parse_metadata", lambda f: {"recording_start_time": starts[f]}
    )
    bounds = (
        (datetime.time(22), datetime.time(2))
        if overnight
        else (datetime.time(10), datetime.time(14))
    )
    assert utils.select_files_in_time_range(iter(["a", "b", "a"]), *bounds, "UTC") == (
        ["a", "a"] if overnight else ["b"]
    )
    assert utils.select_files_in_date_range(
        ["a", "b"],
        datetime.date(2024, 1, 1),
        datetime.date(2024, 1, 1),
        "America/New_York",
    ) == ["a"]


def test_select_missing_and_filename_fallback(monkeypatch):
    monkeypatch.setattr(audio, "parse_metadata", lambda f: {})
    args = (datetime.date(2024, 1, 1), datetime.date(2024, 1, 1), "UTC")
    with pytest.raises(ValueError):
        utils.select_files_in_date_range(
            ["20240101_120000.WAV"], *args, raise_exceptions=True
        )
    assert (
        utils.select_files_in_date_range(["unknown"], *args, raise_exceptions=False)
        == []
    )
    assert utils.select_files_in_date_range(
        ["20240101_120000.WAV"],
        *args,
        parse_timestamp_from_filename=True,
        file_timezone="UTC",
    ) == ["20240101_120000.WAV"]


def test_select_timezone_conflict(monkeypatch):
    monkeypatch.setattr(
        audio,
        "parse_metadata",
        lambda f: {
            "recording_start_time": datetime.datetime(
                2024, 1, 1, tzinfo=datetime.timezone.utc
            )
        },
    )
    with pytest.raises(utils.GetRecordingStartTimeError):
        utils.select_files_in_time_range(
            ["a"],
            datetime.time(0),
            datetime.time(23),
            "UTC",
            file_timezone="America/New_York",
            raise_exceptions=True,
        )


@pytest.mark.parametrize("raise_exceptions", [False, True])
@pytest.mark.parametrize(
    "hour",
    [datetime.datetime(2024, 3, 10, 2, 30), datetime.datetime(2024, 11, 3, 1, 30)],
)
def test_select_dst_failure(monkeypatch, raise_exceptions, hour):
    monkeypatch.setattr(
        audio, "parse_metadata", lambda f: {"recording_start_time": hour}
    )
    options = dict(file_timezone="America/New_York", raise_exceptions=raise_exceptions)
    if raise_exceptions:
        with pytest.raises(utils.GetRecordingStartTimeError) as error:
            utils.select_files_in_time_range(
                ["a"], datetime.time(0), datetime.time(23), "UTC", **options
            )
        assert isinstance(
            error.value.__cause__, (pytz.AmbiguousTimeError, pytz.NonExistentTimeError)
        )
    else:
        assert (
            utils.select_files_in_time_range(
                ["a"], datetime.time(0), datetime.time(23), "UTC", **options
            )
            == []
        )


def test_select_validates_empty_input():
    with pytest.raises(pytz.UnknownTimeZoneError):
        utils.select_files_in_time_range(
            [], datetime.time(0), datetime.time(23), "invalid"
        )


@pytest.fixture()
def gps_metadata_wav_str():
    return "tests/audio/audiomoth_gps.wav"


@pytest.fixture()
def no_timestamp_wav_str():
    return "tests/audio/short_no_metadata.wav"


@pytest.mark.parametrize(
    "fixture_name,recording_date,recording_time",
    [
        ("metadata_wav_str", datetime.date(2020, 4, 4), datetime.time(10, 25)),
        ("gps_metadata_wav_str", datetime.date(2024, 5, 28), datetime.time(10)),
    ],
)
def test_select_file_based_metadata(
    request, fixture_name, recording_date, recording_time
):
    file = request.getfixturevalue(fixture_name)
    assert utils.select_files_in_date_range(
        [file], recording_date, recording_date, "UTC", raise_exceptions=True
    ) == [file]
    other_date = recording_date + datetime.timedelta(days=1)
    assert (
        utils.select_files_in_date_range(
            [file], other_date, other_date, "UTC", raise_exceptions=True
        )
        == []
    )
    assert utils.select_files_in_time_range(
        [file], recording_time, recording_time, "UTC", raise_exceptions=True
    ) == [file]
    assert (
        utils.select_files_in_time_range(
            [file], datetime.time(12), datetime.time(13), "UTC", raise_exceptions=True
        )
        == []
    )


def test_select_file_based_metadata_across_midnight(metadata_wav_str):
    file = metadata_wav_str
    assert utils.select_files_in_date_range(
        [file],
        datetime.date(2020, 4, 5),
        datetime.date(2020, 4, 5),
        "Pacific/Kiritimati",
        raise_exceptions=True,
    ) == [file]
    assert utils.select_files_in_time_range(
        [file],
        datetime.time(22),
        datetime.time(2),
        "Pacific/Kiritimati",
        raise_exceptions=True,
    ) == [file]


def test_select_file_based_missing_timestamp(metadata_wav_str, no_timestamp_wav_str):
    files = [metadata_wav_str, no_timestamp_wav_str, Path(metadata_wav_str)]
    bounds = (datetime.date(2020, 4, 4), datetime.date(2020, 4, 4), "UTC")
    assert utils.select_files_in_date_range(files, *bounds) == [files[0], files[2]]
    with pytest.raises(
        utils.GetRecordingStartTimeError, match=no_timestamp_wav_str
    ) as error:
        utils.select_files_in_date_range(files, *bounds, raise_exceptions=True)
    assert isinstance(error.value.__cause__, ValueError)


def test_select_file_based_filename_fallback(tmp_path, no_timestamp_wav_str):
    file = tmp_path / "20240101_120000.WAV"
    shutil.copyfile(no_timestamp_wav_str, file)
    bounds = (datetime.date(2024, 1, 1), datetime.date(2024, 1, 1), "UTC")
    assert utils.select_files_in_date_range([file], *bounds) == []
    assert utils.select_files_in_date_range(
        [file],
        *bounds,
        parse_timestamp_from_filename=True,
        file_timezone="UTC",
        raise_exceptions=True,
    ) == [file]
    with pytest.raises(utils.GetRecordingStartTimeError) as error:
        utils.select_files_in_date_range(
            [file],
            *bounds,
            parse_timestamp_from_filename=True,
            raise_exceptions=True,
        )
    assert "file_timezone" in str(error.value.__cause__)


def test_select_file_based_metadata_precedes_filename(tmp_path, metadata_wav_str):
    file = tmp_path / "20240101_120000.WAV"
    shutil.copyfile(metadata_wav_str, file)
    assert utils.select_files_in_date_range(
        [file],
        datetime.date(2020, 4, 4),
        datetime.date(2020, 4, 4),
        "UTC",
        parse_timestamp_from_filename=True,
        raise_exceptions=True,
    ) == [file]
