#!/usr/bin/env python3
import pytest
from pathlib import Path
import pandas as pd
import numpy as np

from opensoundscape import annotations
from opensoundscape.annotations import BoxedAnnotations
from opensoundscape.utils import generate_clip_times_df, GetDurationError


@pytest.fixture()
def raven_file():
    return "tests/raven_annots/MSD-0003_20180427_2minstart00.Table.1.selections.txt"


@pytest.fixture()
def raven_file_empty():
    return "tests/raven_annots/EmptyExample.Table.1.selections.txt"


@pytest.fixture()
def saved_raven_file(request):
    path = Path("tests/raven_annots/audio_file.selections.txt")

    # remove this after tests are complete
    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def save_path():
    return Path("tests/raven_annots/")


@pytest.fixture()
def silence_10s_mp3_str():
    return "tests/audio/silence_10s.mp3"


@pytest.fixture()
def rugr_wav_str():
    return "tests/audio/rugr_drum.wav"


@pytest.fixture()
def boxed_annotations():
    df = pd.DataFrame(
        data={
            "audio_file": ["audio_file.wav"] * 3,
            "start_time": [0, 3, 4],
            "end_time": [1, 5, 5],
            "low_f": [0, 500, 1000],
            "high_f": [100, 1200, 1500],
            "annotation": ["a", "b", None],
        }
    )
    return BoxedAnnotations(
        df,
        audio_files=["audio_file.wav"] * 3,
        annotation_files=["audio_file.annotations.txt"] * 3,
    )


@pytest.fixture()
def boxed_annotations_zero_len():
    df = pd.DataFrame(
        data={
            "audio_file": ["audio_file.wav"] * 3,
            "start_time": [0, 3, 4],
            "end_time": [0, 3, 4],
            "low_f": [0, 500, 1000],
            "high_f": [100, 1200, 1500],
            "annotation": ["a", "b", None],
        }
    )
    return BoxedAnnotations(df)


def test_load_raven_annotations(raven_file):
    ba = BoxedAnnotations.from_raven_files([raven_file])
    assert len(ba.df) == 10
    assert set(ba.df["annotation"]) == {"WOTH", "EATO", "LOWA", np.nan}

    def isnan(x):
        return x != x

    assert isnan(ba.df["audio_file"].values[0])


def test_load_raven_annotations_w_audio(raven_file):
    ba = BoxedAnnotations.from_raven_files([raven_file], ["audio_path"])
    assert set(ba.df["annotation"]) == {"WOTH", "EATO", "LOWA", np.nan}
    assert ba.df["audio_file"].values[0] == "audio_path"


def test_load_raven_no_annotation_column(raven_file):
    a = BoxedAnnotations.from_raven_files([raven_file], annotation_column_idx=None)
    # we should now have a dataframe with a column "Species"
    assert len(a.df) == 10
    assert set(a.df["Species"]) == {"WOTH", "EATO", "LOWA", np.nan}


def test_load_raven_annotation_column_name(raven_file):
    # specify the name of the annotation column
    a = BoxedAnnotations.from_raven_files(
        [raven_file], annotation_column_name="Species"
    )
    assert a.df["annotation"].values[0] == "WOTH"

    # use a different column
    a = BoxedAnnotations.from_raven_files([raven_file], annotation_column_name="View")
    assert a.df["annotation"].values[0] == "Spectrogram 1"

    # use a column that doesn't exist: annotations should be nan
    a = BoxedAnnotations.from_raven_files(
        [raven_file], annotation_column_name="notacolumn"
    )
    assert a.df["annotation"].values[0] != a.df["annotation"].values[0]


def test_load_raven_annotations_empty(raven_file_empty):
    a = BoxedAnnotations.from_raven_files([raven_file_empty])
    assert len(a.df) == 0


def test_load_raven_annotations_different_columns(raven_file, raven_file_empty):
    # keep all extra columns
    ba = BoxedAnnotations.from_raven_files(
        [raven_file, raven_file_empty], keep_extra_columns=True
    )
    assert "distance" in list(ba.df.columns)
    assert "type" in list(ba.df.columns)
    assert "annotation_file" in list(ba.df.columns)

    # keep one extra column
    ba = BoxedAnnotations.from_raven_files(
        [raven_file, raven_file_empty], keep_extra_columns=["distance"]
    )
    assert "distance" in list(ba.df.columns)
    assert not "type" in list(ba.df.columns)
    # this would fail before #737 was resolved
    assert "annotation_file" in list(ba.df.columns)
    # check for #769

    # keep no extra column
    ba = BoxedAnnotations.from_raven_files(
        [raven_file, raven_file_empty], keep_extra_columns=False
    )
    assert not "distance" in list(ba.df.columns)
    assert not "type" in list(ba.df.columns)
    assert "annotation_file" in list(ba.df.columns)


def test_to_raven_files(boxed_annotations, saved_raven_file):
    """note: assumes raven file will be named audio_file.annotations.txt"""
    assert not saved_raven_file.exists()
    boxed_annotations.to_raven_files(saved_raven_file.parent)
    assert saved_raven_file.exists()


def test_subset(boxed_annotations):
    subset = boxed_annotations.subset(["a", None])
    assert len(subset.df) == 2
    # should retain .audio_files and .annotation_files
    assert subset.audio_files == boxed_annotations.audio_files
    assert subset.annotation_files == boxed_annotations.annotation_files


def test_subset_to_nan(raven_file):
    a = BoxedAnnotations.from_raven_files([raven_file])
    assert len(a.subset([np.nan]).df) == 1


def test_trim(boxed_annotations):
    trimmed = boxed_annotations.trim(0.5, 3.5, edge_mode="trim")
    assert len(trimmed.df) == 2
    assert np.max(trimmed.df["end_time"]) == 3.0
    assert np.min(trimmed.df["start_time"]) == 0.0

    # should retain .audio_files and .annotation_files
    assert trimmed.audio_files == boxed_annotations.audio_files
    assert trimmed.annotation_files == boxed_annotations.annotation_files


def test_trim_keep(boxed_annotations):
    trimmed = boxed_annotations.trim(0.5, 3.5, edge_mode="keep")
    assert len(trimmed.df) == 2
    assert np.max(trimmed.df["end_time"]) == 4.5
    assert np.min(trimmed.df["start_time"]) == -0.5


def test_trim_remove(boxed_annotations):
    trimmed = boxed_annotations.trim(0.5, 3.5, edge_mode="remove")
    assert len(trimmed.df) == 0


def test_bandpass(boxed_annotations):
    bandpassed = boxed_annotations.bandpass(600, 1400, edge_mode="trim")
    assert len(bandpassed.df) == 2
    assert np.max(bandpassed.df["high_f"]) == 1400
    assert np.min(bandpassed.df["low_f"]) == 600
    # should retain .audio_files and .annotation_files
    assert bandpassed.audio_files == boxed_annotations.audio_files
    assert bandpassed.annotation_files == boxed_annotations.annotation_files


def test_bandpass_keep(boxed_annotations):
    bandpassed = boxed_annotations.bandpass(600, 1400, edge_mode="keep")
    assert len(bandpassed.df) == 2
    assert np.max(bandpassed.df["high_f"]) == 1500
    assert np.min(bandpassed.df["low_f"]) == 500


def test_bandpass_remove(boxed_annotations):
    bandpassed = boxed_annotations.bandpass(600, 1400, edge_mode="remove")
    assert len(bandpassed.df) == 0


def test_unique_labels(boxed_annotations):
    assert set(boxed_annotations.unique_labels()) == set(["a", "b"])


def test_global_multi_hot_labels(boxed_annotations):
    assert boxed_annotations.global_multi_hot_labels(classes=["a", "b", "c"]) == [
        1,
        1,
        0,
    ]


def test_multi_hot_labels_like(boxed_annotations):
    clip_df = generate_clip_times_df(5, clip_duration=1.0, clip_overlap=0)
    clip_df["audio_file"] = "audio_file.wav"
    clip_df = clip_df.set_index(["audio_file", "start_time", "end_time"])
    labels = boxed_annotations.multi_hot_labels_like(
        clip_df, class_subset=["a"], min_label_overlap=0.25
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())


def test_multi_hot_labels_like_overlap(boxed_annotations):
    clip_df = generate_clip_times_df(3, clip_duration=1.0, clip_overlap=0.5)
    clip_df["audio_file"] = "audio_file.wav"
    clip_df = clip_df.set_index(["audio_file", "start_time", "end_time"])
    labels = boxed_annotations.multi_hot_labels_like(
        clip_df, class_subset=["a"], min_label_overlap=0.25
    )
    assert np.array_equal(labels.values, np.array([[1, 1, 0, 0, 0]]).transpose())


def test_multi_hot_clip_labels(boxed_annotations):
    labels = boxed_annotations.multi_hot_clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0.25,
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())


def test_multi_hot_clip_labels_get_duration(boxed_annotations, silence_10s_mp3_str):
    """should get duration of audio files if full_duration is None"""
    boxed_annotations.df["audio_file"] = [silence_10s_mp3_str] * len(
        boxed_annotations.df
    )
    labels = boxed_annotations.multi_hot_clip_labels(
        full_duration=None,
        clip_duration=2.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0.25,
        audio_files=[silence_10s_mp3_str],
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())


def test_multi_hot_clip_labels_exception(boxed_annotations):
    """raises GetDurationError because file length cannot be determined
    and full_duration is None
    """
    boxed_annotations.audio_files = ["non existant file"]
    with pytest.raises(GetDurationError):
        labels = boxed_annotations.multi_hot_clip_labels(
            full_duration=None,
            clip_duration=2.0,
            clip_overlap=0,
            class_subset=["a"],
            min_label_overlap=0.25,
        )


def test_multi_hot_clip_labels_overlap(boxed_annotations):
    labels = boxed_annotations.multi_hot_clip_labels(
        full_duration=3,
        clip_duration=1.0,
        clip_overlap=0.5,
        class_subset=["a"],
        min_label_overlap=0.25,
    )
    assert np.array_equal(labels.values, np.array([[1, 1, 0, 0, 0]]).transpose())


def test_convert_labels(boxed_annotations):
    boxed_annotations1 = boxed_annotations.convert_labels({"a": "c"})
    assert set(boxed_annotations1.df["annotation"]) == {"b", "c", None}
    # should retain properties, issue #916
    assert boxed_annotations1.audio_files == boxed_annotations.audio_files


def test_convert_labels_df(boxed_annotations):
    df = pd.DataFrame({0: ["a"], 1: ["c"]})
    boxed_annotations = boxed_annotations.convert_labels(df)
    assert set(boxed_annotations.df["annotation"]) == {"b", "c", None}


def test_convert_labels_empty(boxed_annotations):
    boxed_annotations = boxed_annotations.convert_labels({})
    assert set(boxed_annotations.df["annotation"]) == {"a", "b", None}


def test_convert_labels_wrong_type(boxed_annotations):
    df = [["a", "b", "c"], ["a", "b", "d"]]
    with pytest.raises(TypeError):
        boxed_annotations = boxed_annotations.convert_labels(df)


def test_multi_hot_labels_on_time_interval(boxed_annotations):
    a = annotations.multi_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0,
        end_time=3.5,
        min_label_overlap=0.25,
        class_subset=["a", "b"],
    )
    assert a["a"] == 1 and a["b"] == 1

    a = annotations.multi_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0,
        end_time=3.5,
        min_label_overlap=0.75,
        class_subset=["a", "b"],
    )
    assert a["a"] == 1 and a["b"] == 0


def test_multi_hot_labels_on_time_interval_fractional(boxed_annotations):
    """test min_label_fraction use cases"""
    # too short but satisfies fraction
    a = annotations.multi_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0.4,
        end_time=3,
        min_label_overlap=2,
        min_label_fraction=0.5,
        class_subset=["a"],
    )
    assert a["a"] == 1

    # too short and not enough for fraction
    a = annotations.multi_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0.4,
        end_time=3,
        min_label_overlap=2,
        min_label_fraction=0.9,
        class_subset=["a"],
    )
    assert a["a"] == 0

    # long enough, although less than fraction
    a = annotations.multi_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0.4,
        end_time=3,
        min_label_overlap=0.5,
        min_label_fraction=0.9,
        class_subset=["a"],
    )
    assert a["a"] == 1


def test_categorical_to_multi_hot():
    cat_labels = [["a", "b"], ["a", "c"]]
    multi_hot, classes = annotations.categorical_to_multi_hot(
        cat_labels, class_subset=["a", "b", "c", "d"]
    )
    assert set(classes) == {"a", "b", "c", "d"}
    assert multi_hot.tolist() == [[1, 1, 0, 0], [1, 0, 1, 0]]

    # without passing classes list:
    multi_hot, classes = annotations.categorical_to_multi_hot(cat_labels)
    assert set(classes) == {"a", "b", "c"}


def test_multi_hot_to_categorical():
    classes = ["a", "b", "c"]
    multi_hot = [[0, 0, 1], [1, 1, 1]]
    cat_labels = annotations.multi_hot_to_categorical(multi_hot, classes)
    assert list(cat_labels) == [["c"], ["a", "b", "c"]]


def test_multi_hot_to_categorical_and_back():
    classes = ["a", "b", "c"]
    multi_hot = [[0, 0, 1], [1, 1, 1]]
    cat_labels = annotations.multi_hot_to_categorical(multi_hot, classes)
    multi_hot2, classes2 = annotations.categorical_to_multi_hot(cat_labels, classes)

    assert np.array_equal(multi_hot, multi_hot2)
    assert np.array_equal(classes, classes2)


# test robustness of raven methods for empty annotation file
def test_raven_annotation_methods_empty(raven_file_empty):
    a = BoxedAnnotations.from_raven_files([raven_file_empty])

    a.trim(0, 5)
    a.bandpass(0, 11025)
    assert len(a.df) == 0

    # test with random parameters to generate clip dataframe
    clip_df = generate_clip_times_df(
        full_duration=10,
        clip_duration=2,
    )
    clip_df["audio_file"] = "a"
    clip_df = clip_df.set_index(["audio_file", "start_time", "end_time"])

    # class_subset = None: keep all
    labels_df = a.multi_hot_labels_like(
        clip_df,
        class_subset=None,
        min_label_overlap=0.25,
    )

    assert (labels_df.reset_index() == clip_df.reset_index()).all().all()

    # classes = subset
    labels_df = a.multi_hot_labels_like(
        clip_df,
        class_subset=["Species1", "Species2"],
        min_label_overlap=0.25,
    )

    assert len(labels_df) == len(clip_df)
    assert (labels_df.columns == ["Species1", "Species2"]).all()


def test_methods_on_zero_length_annotations(boxed_annotations_zero_len):
    # time range is inclusive on left bound; includes 0
    trimmed = boxed_annotations_zero_len.trim(0, 1)
    assert len(trimmed.df == 1)

    # time range is exclusive on right bound, excludes 3
    trimmed = boxed_annotations_zero_len.trim(2, 3)
    assert len(trimmed.df) == 0

    bandpassed = boxed_annotations_zero_len.bandpass(0, 1000)
    assert len(bandpassed.df == 1)

    filtered = boxed_annotations_zero_len.subset(["a"])
    assert len(filtered.df == 1)


def test_multi_hot_clip_labels_with_empty_annotation_file(
    raven_file_empty, silence_10s_mp3_str, raven_file, rugr_wav_str
):
    """test that multi_hot_clip_labels works with empty annotation file

    it should return a dataframe with rows for each clip and 0s for all labels
    """
    boxed_annotations = BoxedAnnotations.from_raven_files(
        [raven_file_empty], [silence_10s_mp3_str]
    )

    small_label_df = boxed_annotations.multi_hot_clip_labels(
        full_duration=None,
        clip_duration=4,
        clip_overlap=2,
        min_label_overlap=0.1,
        class_subset=["EATO", "REVI"],
        final_clip=None,
    )

    # 10 s clips has start times at 0,2,4,6 s
    assert len(small_label_df) == 4
    assert (small_label_df == 0).all().all()

    # should also work when concatenating empty and non-empty annotation files
    boxed_annotations = BoxedAnnotations.from_raven_files(
        [raven_file_empty, raven_file], [silence_10s_mp3_str, rugr_wav_str]
    )

    small_label_df = boxed_annotations.multi_hot_clip_labels(
        full_duration=None,
        clip_duration=4,
        clip_overlap=2,
        min_label_overlap=0.1,
        class_subset=["EATO", "REVI"],
        final_clip=None,
    )
    # should have clip entries for both clips
    assert len(small_label_df) == 8


def test_to_raven_files_raises_if_no_audio_files(raven_file, save_path):
    # raises ValueError if no audio_files is provided and self.audio_files is none
    with pytest.raises(ValueError):
        # don't save to a path with a .finalizer(), because the finalizer will complain
        # if the file isn't actually created
        boxed_annotations = BoxedAnnotations.from_raven_files([raven_file])
        boxed_annotations.to_raven_files(save_path)


def test_warn_if_file_wont_get_raven_output(raven_file, saved_raven_file):
    # should also work when concatenating empty and non-empty annotation files
    boxed_annotations = BoxedAnnotations.from_raven_files([raven_file], ["path1"])
    with pytest.warns(UserWarning):
        boxed_annotations.to_raven_files(
            saved_raven_file.parent, audio_files=["audio_file"]
        )


def test_assert_audio_files_annotation_files_match():
    with pytest.raises(AssertionError):
        BoxedAnnotations.from_raven_files(["path"], ["a", "b"])


def test_assert_audio_files_annotation_files_empty():
    with pytest.raises(AssertionError):
        BoxedAnnotations.from_raven_files([], [])


def test_from_raven_files(raven_file):
    ba = BoxedAnnotations.from_raven_files([raven_file], ["path1"])
    assert ba.annotation_files[0] == raven_file


def test_from_raven_files_pathlib(raven_file):
    ba = BoxedAnnotations.from_raven_files([Path(raven_file)], [Path("path1")])
    assert str(ba.annotation_files[0]) == raven_file


def test_from_raven_files_one_path(raven_file):
    """now works passing str or Path rather than list"""
    ba = BoxedAnnotations.from_raven_files(raven_file, ["path1"])
    assert ba.annotation_files[0] == raven_file
    assert len(ba.annotation_files) == 1
    ba = BoxedAnnotations.from_raven_files(Path(raven_file), ["path1"])
    assert str(ba.annotation_files[0]) == raven_file
    assert len(ba.annotation_files) == 1


def test_from_raven_files_one_audio_file(raven_file):
    """now works passing str or Path rather than list"""
    ba = BoxedAnnotations.from_raven_files(raven_file, "path1")
    assert ba.audio_files[0] == "path1"
    assert len(ba.audio_files) == 1
    ba = BoxedAnnotations.from_raven_files(Path(raven_file), Path("path1"))
    assert str(ba.audio_files[0]) == "path1"
    assert len(ba.audio_files) == 1
