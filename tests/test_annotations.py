#!/usr/bin/env python3
import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import crowsetta

from opensoundscape import annotations
from opensoundscape.annotations import BoxedAnnotations
from opensoundscape.utils import generate_clip_times_df, GetDurationError


@pytest.fixture()
def raven_file():
    return "tests/raven_annots/MSD-0003_20180427_2minstart00.Table.1.selections.txt"


@pytest.fixture()
def raven_file_Annotation_col():
    return "tests/raven_annots/raven_with_Annotation_col.txt"


@pytest.fixture()
def audio_2min():
    return "tests/audio/MSD-0003_20180427_2minstart00.wav"


@pytest.fixture()
def raven_file_empty():
    return "tests/raven_annots/EmptyExample.Table.1.selections.txt"


@pytest.fixture()
def audio_silence():
    return "tests/audio/silence_10s.mp3"


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
def saved_csv(request):
    path = "tests/csvs/labels.csv"

    def fin():
        Path(path).unlink()

    request.addfinalizer(fin)

    return path


@pytest.fixture()
def silence_10s_mp3_str():
    return "tests/audio/silence_10s.mp3"


@pytest.fixture()
def rugr_wav_str():
    return "tests/audio/rugr_drum.wav"


@pytest.fixture()
def labels_df():
    return pd.DataFrame(
        {
            "file": ["audio_file.wav"] * 3,
            "start_time": [0, 3, 6],
            "end_time": [3, 6, 9],
            "labels": [["a", "b"], ["b", "c"], ["a", "c"]],
        }
    )


@pytest.fixture()
def labels_df_int():
    return pd.DataFrame(
        {
            "file": ["audio_file.wav"] * 3,
            "start_time": [0, 3, 6],
            "end_time": [3, 6, 9],
            "labels": [[0, 1], [1, 2], [0, 2]],
        }
    )


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
        audio_files=["audio_file.wav"],
        annotation_files=["audio_file.annotations.txt"],
    )


@pytest.fixture()
def boxed_annotations_2_files():
    df = pd.DataFrame(
        data={
            "audio_file": ["audio_file.wav"] * 2 + ["audio2.wav"],
            "annotation_file": ["ann.txt"] * 2 + ["ann2.txt"],
            "start_time": [0, 3, 4],
            "end_time": [1, 5, 5],
            "low_f": [0, 500, 1000],
            "high_f": [100, 1200, 1500],
            "annotation": ["a", "b", None],
        }
    )
    return BoxedAnnotations(
        df,
        audio_files=["audio_file.wav", "audio2.wav"],
        annotation_files=["ann.txt", "ann2.txt"],
    )


@pytest.fixture()
def boxed_annotations_double_ann():
    df = pd.DataFrame(
        data={
            "audio_file": ["audio_file.wav"] * 2,
            "annotation_file": ["ann.txt"] * 2,
            "start_time": [0, 1],
            "end_time": [3, 2],
            "low_f": [0, 500],
            "high_f": [100, 1200],
            "annotation": ["a", "a"],
        }
    )
    return BoxedAnnotations(df, audio_files=["audio_file.wav"])


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


def test_init_boxed_annotations_with_no_df():
    ba = BoxedAnnotations(df=None)  # init without passing df
    assert len(ba.df) == 0
    assert list(ba.df.columns) == BoxedAnnotations._standard_cols


def test_init_boxed_annotations_with_only_reqd_cols():
    """creates df with nan in other standard columns"""
    df = pd.DataFrame({"annotation": ["a"], "start_time": [0], "end_time": [1]})
    ba = BoxedAnnotations(df)
    assert len(ba.df) == 1


def test_load_raven_annotations(raven_file):
    ba = BoxedAnnotations.from_raven_files([raven_file], "Species")
    assert len(ba.df) == 10
    assert set(ba.df["annotation"]) == {"WOTH", "EATO", "LOWA", np.nan}

    def isnan(x):
        return x != x

    assert isnan(ba.df["audio_file"].values[0])


def test_concat_boxed_annotations(boxed_annotations):
    joined = BoxedAnnotations.concat([boxed_annotations] * 3)
    assert len(joined.df) == 9
    assert len(joined.audio_files) == 3
    assert len(joined.annotation_files) == 3

    # handles scenario where audio_files and/or annotation_files are None
    boxed_annotations.annotation_files = None
    boxed_annotations.audio_files = None
    joined = BoxedAnnotations.concat([boxed_annotations] * 3)
    assert len(joined.df) == 9
    assert joined.audio_files is None
    assert joined.annotation_files is None


def test_load_raven_annotations_w_audio(raven_file):
    ba = BoxedAnnotations.from_raven_files([raven_file], "Species", ["audio_path"])
    assert set(ba.df["annotation"]) == {"WOTH", "EATO", "LOWA", np.nan}
    assert ba.df["audio_file"].values[0] == "audio_path"


def test_load_raven_no_annotation_column(raven_file):
    a = BoxedAnnotations.from_raven_files([raven_file], annotation_column=None)
    # we should now have a dataframe with a column "Species"
    assert len(a.df) == 10
    assert set(a.df["Species"]) == {"WOTH", "EATO", "LOWA", np.nan}
    assert a.df["annotation"].isna().all()


def test_load_raven_annotation_column_name(raven_file):
    # specify the name of the annotation column
    a = BoxedAnnotations.from_raven_files([raven_file], annotation_column="Species")
    assert a.df["annotation"].values[0] == "WOTH"

    # use a different column
    a = BoxedAnnotations.from_raven_files([raven_file], annotation_column="View")
    assert a.df["annotation"].values[0] == "Spectrogram 1"

    with pytest.raises(KeyError):
        # using a column name that doesn't exist shoud raise an error
        a = BoxedAnnotations.from_raven_files(
            [raven_file], annotation_column="notacolumn"
        )

    # now try integer index values
    a = BoxedAnnotations.from_raven_files([raven_file], annotation_column=7)
    assert a.df["annotation"].values[0] == "WOTH"

    # use different column number
    a = BoxedAnnotations.from_raven_files([raven_file], annotation_column=1)
    assert a.df["annotation"].values[0] == "Spectrogram 1"

    # try using an out of bounds number - raises an exception
    with pytest.raises(IndexError):
        a = BoxedAnnotations.from_raven_files([raven_file], annotation_column=25)
    with pytest.raises(IndexError):
        a = BoxedAnnotations.from_raven_files([raven_file], annotation_column=-1)


def test_from_raven_files_list_of_annotation_column(
    raven_file, raven_file_Annotation_col
):
    ba = BoxedAnnotations.from_raven_files(
        [raven_file, raven_file_Annotation_col],
        annotation_column=["Species", "Annotation"],
    )
    assert "CSWA" in ba.unique_labels() and "WOTH" in ba.unique_labels()

    # also allowed to be a tuple
    ba = BoxedAnnotations.from_raven_files(
        [raven_file, raven_file_Annotation_col],
        annotation_column=("Species", "Annotation"),
    )
    assert "CSWA" in ba.unique_labels() and "WOTH" in ba.unique_labels()

    # raises an exception if no matching column is found
    with pytest.raises(KeyError):
        ba = BoxedAnnotations.from_raven_files(
            [raven_file, raven_file_Annotation_col],
            annotation_column=["Species", "notacolumn"],
        )

    # raises an exception if multiple matching columns are found
    with pytest.raises(KeyError):
        ba = BoxedAnnotations.from_raven_files(
            [raven_file, raven_file_Annotation_col],
            annotation_column=["Species", "Selection"],
        )


def test_load_raven_annotations_empty(raven_file_empty):
    a = BoxedAnnotations.from_raven_files([raven_file_empty], None)
    assert len(a.df) == 0


def test_load_raven_annotations_different_columns(raven_file, raven_file_empty):
    # keep all extra columns
    ba = BoxedAnnotations.from_raven_files(
        [raven_file, raven_file_empty], None, keep_extra_columns=True
    )
    assert "distance" in list(ba.df.columns)
    assert "type" in list(ba.df.columns)
    assert "annotation_file" in list(ba.df.columns)

    # keep one extra column
    ba = BoxedAnnotations.from_raven_files(
        [raven_file, raven_file_empty], None, keep_extra_columns=["distance"]
    )
    assert "distance" in list(ba.df.columns)
    assert not "type" in list(ba.df.columns)
    # this would fail before #737 was resolved
    assert "annotation_file" in list(ba.df.columns)
    # check for #769

    # keep no extra column
    ba = BoxedAnnotations.from_raven_files(
        [raven_file, raven_file_empty], None, keep_extra_columns=False
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
    a = BoxedAnnotations.from_raven_files([raven_file], "Species")
    assert len(a.subset([np.nan]).df) == 1


def test_subset_all_nan_to_nan(raven_file):
    # test behavior where entire row is nan - previously was fragile
    a = BoxedAnnotations.from_raven_files([raven_file], None)
    assert len(a.subset([np.nan]).df) == len(a.df)


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


def test_labels_on_index(boxed_annotations):
    clip_df = generate_clip_times_df(5, clip_duration=1.0, clip_overlap=0)
    clip_df["file"] = "audio_file.wav"
    clip_df = clip_df.set_index(["file", "start_time", "end_time"])

    # test multihot return type
    labels = boxed_annotations.labels_on_index(
        clip_df, class_subset=["a"], min_label_overlap=0.25, return_type="multihot"
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())

    # test integers return type
    labels, classes = boxed_annotations.labels_on_index(
        clip_df, class_subset=["a"], min_label_overlap=0.25, return_type="integers"
    )
    assert labels.labels.to_list() == [[0], [], [], [], []]
    assert classes == ["a"]

    # test classes return type
    labels, classes = boxed_annotations.labels_on_index(
        clip_df, class_subset=["a"], min_label_overlap=0.25, return_type="classes"
    )
    assert labels.labels.to_list() == [["a"], [], [], [], []]

    # test CategoricalLabels return type
    labels = boxed_annotations.labels_on_index(
        clip_df,
        class_subset=["a"],
        min_label_overlap=0.25,
        return_type="CategoricalLabels",
    )
    assert isinstance(labels, annotations.CategoricalLabels)
    assert list(labels.multihot_dense) == [[1], [0], [0], [0], [0]]


def test_labels_on_index_no_overlap(boxed_annotations):
    # check it does not fail if no annotations overlap with any of the clip_df times
    clip_df = pd.DataFrame.from_dict(
        {
            "file": ["audio_file.wav"] * 2,
            "start_time": [50, 60],  # after all the annotations
            "end_time": [60, 70],
        }
    )
    clip_df = clip_df.set_index(["file", "start_time", "end_time"])
    labels = boxed_annotations.labels_on_index(
        clip_df, class_subset=["a"], min_label_overlap=0.25
    )
    assert np.array_equal(labels.values, np.array([[0, 0]]).transpose())


def test_labels_on_index_overlap(boxed_annotations):
    clip_df = generate_clip_times_df(3, clip_duration=1.0, clip_overlap=0.5)
    clip_df["audio_file"] = "audio_file.wav"
    clip_df = clip_df.set_index(["audio_file", "start_time", "end_time"])
    labels = boxed_annotations.labels_on_index(
        clip_df, class_subset=["a"], min_label_overlap=0.25
    )
    assert np.array_equal(labels.values, np.array([[1, 1, 0, 0, 0]]).transpose())


def test_clip_labels_with_audio_file(
    raven_file, audio_2min, raven_file_empty, audio_silence
):
    """test that clip_labels works properly with multiple audio+raven files

    checks that Issue #1061 is resolved
    """
    boxed_annotations = BoxedAnnotations.from_raven_files(
        raven_files=[raven_file, raven_file_empty],
        audio_files=[audio_2min, audio_silence],
        annotation_column="Species",
    )
    labels = boxed_annotations.clip_labels(
        full_duration=None, clip_duration=5, clip_overlap=0, min_label_overlap=0
    )
    # should get back 2 min & 10 s audio file labels for 5s clips
    assert len(labels) == 24 + 2
    # no label on silent audio!
    assert labels.head(0).sum().sum() == 0
    # check for correct clip labels
    assert np.array_equal(
        labels.head(4).values,
        np.array(
            [
                [True, True, False],
                [True, True, False],
                [True, True, True],
                [False, True, False],
            ]
        ),
    )
    # no labels after 20 seconds in 2 min audio or in empty audio
    assert labels.tail(-4).sum().sum() == 0


def test_clip_labels(boxed_annotations):
    # test "multihot" return type
    labels = boxed_annotations.clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0.25,
        return_type="multihot",
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())

    # test "integers" return type
    labels, classes = boxed_annotations.clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0.25,
        return_type="integers",
    )
    assert labels.labels.to_list() == [[0], [], [], [], []]
    assert classes == ["a"]

    # test "classes" return type
    labels, classes = boxed_annotations.clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0.25,
        return_type="classes",
    )
    assert labels.labels.to_list() == [["a"], [], [], [], []]
    assert classes == ["a"]

    # test "CategoricalLabels" return type
    labels = boxed_annotations.clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0.25,
        return_type="CategoricalLabels",
    )
    assert isinstance(labels, annotations.CategoricalLabels)
    assert list(labels.multihot_dense) == [[1], [0], [0], [0], [0]]


def test_clip_labels_overlap_fraction(boxed_annotations):
    # test that min_label_fraction argument works as expected.
    # expected behavior is that all clips with at least 50% are labeled, even if
    # the time overlap is less than the min_label_overlap

    labels = boxed_annotations.clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=50,  # longer than any clip. NO clips should be labeled
        min_label_fraction=0.5,  # means that any clip with at least 50% overlap will be labeled
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())


def test_clip_labels_no_double_count(boxed_annotations_double_ann):
    # test that labels are not double counted
    labels = boxed_annotations_double_ann.clip_labels(
        full_duration=10,
        clip_duration=5.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0,
    )
    assert np.array_equal(labels.values, np.array([[1, 0]]).transpose())


def test_clip_labels_count_duplicate(boxed_annotations_double_ann):
    # test that labels are included multiple times when keep_duplicates=True
    labels, classes = boxed_annotations_double_ann.clip_labels(
        full_duration=10,
        clip_duration=5.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0,
        keep_duplicates=True,
        return_type="classes",
    )
    assert labels["labels"].to_list() == [["a", "a"], []]


def test_clip_labels_no_overlaps(boxed_annotations):
    # confirm that no annotations are made if the required overlap is not met
    labels = boxed_annotations.clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=50,  # longer than any clip. NO clips should be labeled
    )
    assert np.array_equal(labels.values, np.array([[0, 0, 0, 0, 0]]).transpose())


def test_clip_labels_overlap_fraction(boxed_annotations):
    # test that min_label_fraction argument works as expected.
    # expected behavior is that all clips with at least 50% are labeled, even if
    # the time overlap is less than the min_label_overlap

    labels = boxed_annotations.clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=50,  # longer than any clip. NO clips should be labeled
        min_label_fraction=0.5,  # means that any clip with at least 50% overlap will be labeled
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())


def test_clip_labels_no_overlaps(boxed_annotations):
    # confirm that no annotations are made if the required overlap is not met
    labels = boxed_annotations.clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=50,  # longer than any clip. NO clips should be labeled
    )
    assert np.array_equal(labels.values, np.array([[0, 0, 0, 0, 0]]).transpose())


def test_clip_labels_get_duration(boxed_annotations, silence_10s_mp3_str):
    """should get duration of audio files if full_duration is None"""
    boxed_annotations.df["audio_file"] = [silence_10s_mp3_str] * len(
        boxed_annotations.df
    )
    labels = boxed_annotations.clip_labels(
        full_duration=None,
        clip_duration=2.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0.25,
        audio_files=[silence_10s_mp3_str],
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())


def test_clip_labels_exception(boxed_annotations):
    """raises GetDurationError because file length cannot be determined
    and full_duration is None
    """
    boxed_annotations.audio_files = ["non existent file"]
    with pytest.raises(GetDurationError):
        labels = boxed_annotations.clip_labels(
            full_duration=None,
            clip_duration=2.0,
            clip_overlap=0,
            class_subset=["a"],
            min_label_overlap=0.25,
        )


def test_clip_labels_overlap(boxed_annotations):
    labels = boxed_annotations.clip_labels(
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


def test_categorical_to_multi_hot():
    cat_labels = [["a", "b"], ["a", "c"]]
    multi_hot, classes = annotations.categorical_to_multi_hot(
        cat_labels, classes=["a", "b", "c", "d"]
    )
    assert set(classes) == {"a", "b", "c", "d"}
    assert multi_hot.tolist() == [[1, 1, 0, 0], [1, 0, 1, 0]]

    # without passing classes list:
    multi_hot, classes = annotations.categorical_to_multi_hot(cat_labels)
    assert set(classes) == {"a", "b", "c"}


def test_categorical_to_multi_hot_sparse():
    cat_labels = [[], ["a", "b"], [], ["c", "a"]]
    multi_hot_sparse, classes = annotations.categorical_to_multi_hot(
        cat_labels, classes=["a", "b", "c", "d"], sparse=True
    )
    assert set(classes) == {"a", "b", "c", "d"}
    assert multi_hot_sparse.todense().tolist() == [
        [0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 1, 0],
    ]


def test_multi_hot_to_categorical():
    classes = ["a", "b", "c"]
    multi_hot = [[0, 0, 1], [1, 1, 1]]
    cat_labels = annotations.multi_hot_to_categorical(multi_hot, classes)
    assert list(cat_labels) == [["c"], ["a", "b", "c"]]


def test_multi_hot_sparse_to_categorical():
    from scipy.sparse import csr_matrix

    cat_labels = [[], ["a", "b"], [], ["c", "a"]]
    multi_hot_sparse, classes = annotations.categorical_to_multi_hot(
        cat_labels, classes=["a", "b", "c", "d", "e"], sparse=True
    )
    cat_labels_new = annotations.multi_hot_to_categorical(multi_hot_sparse, classes)

    # doesn't retain order, just set composition
    for l0, l1 in zip(cat_labels, cat_labels_new):
        assert set(l0) == set(l1)


def test_multi_hot_to_categorical_and_back():
    classes = ["a", "b", "c"]
    multi_hot = [[0, 0, 1], [1, 1, 1]]
    cat_labels = annotations.multi_hot_to_categorical(multi_hot, classes)
    multi_hot2, classes2 = annotations.categorical_to_multi_hot(cat_labels, classes)

    assert np.array_equal(multi_hot, multi_hot2)
    assert np.array_equal(classes, classes2)


# test robustness of raven methods for empty annotation file
def test_raven_annotation_methods_empty(raven_file_empty):
    a = BoxedAnnotations.from_raven_files([raven_file_empty], None)

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
    labels_df = a.labels_on_index(
        clip_df,
        class_subset=None,
        min_label_overlap=0.25,
    )

    assert (labels_df.reset_index() == clip_df.reset_index()).all().all()

    # classes = subset
    labels_df = a.labels_on_index(
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


def test_clip_labels_with_empty_annotation_file(
    raven_file_empty, silence_10s_mp3_str, raven_file, rugr_wav_str
):
    """test that clip_labels works with empty annotation file

    it should return a dataframe with rows for each clip and 0s for all labels
    """
    boxed_annotations = BoxedAnnotations.from_raven_files(
        [raven_file_empty], None, [silence_10s_mp3_str]
    )

    small_label_df = boxed_annotations.clip_labels(
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
        [raven_file_empty, raven_file], None, [silence_10s_mp3_str, rugr_wav_str]
    )

    small_label_df = boxed_annotations.clip_labels(
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
        boxed_annotations = BoxedAnnotations.from_raven_files([raven_file], None)
        boxed_annotations.to_raven_files(save_path)


def test_warn_if_file_wont_get_raven_output(raven_file, saved_raven_file):
    # should also work when concatenating empty and non-empty annotation files
    boxed_annotations = BoxedAnnotations.from_raven_files([raven_file], None, ["path1"])
    with pytest.warns(UserWarning):
        boxed_annotations.to_raven_files(
            saved_raven_file.parent, audio_files=["audio_file"]
        )


def test_assert_audio_files_annotation_files_match():
    with pytest.raises(AssertionError):
        BoxedAnnotations.from_raven_files(["path"], None, ["a", "b"])


def test_assert_audio_files_annotation_files_empty():
    with pytest.raises(AssertionError):
        BoxedAnnotations.from_raven_files([], None, [])


def test_from_raven_files(raven_file):
    ba = BoxedAnnotations.from_raven_files([raven_file], None, ["path1"])
    assert ba.annotation_files[0] == raven_file


def test_from_raven_files_pathlib(raven_file):
    ba = BoxedAnnotations.from_raven_files([Path(raven_file)], None, [Path("path1")])
    assert str(ba.annotation_files[0]) == raven_file


def test_from_raven_files_one_path(raven_file):
    """now works passing str or Path rather than list"""
    ba = BoxedAnnotations.from_raven_files(raven_file, None, ["path1"])
    assert ba.annotation_files[0] == raven_file
    assert len(ba.annotation_files) == 1
    ba = BoxedAnnotations.from_raven_files(Path(raven_file), None, ["path1"])
    assert str(ba.annotation_files[0]) == raven_file
    assert len(ba.annotation_files) == 1


def test_from_raven_files_one_audio_file(raven_file):
    """now works passing str or Path rather than list"""
    ba = BoxedAnnotations.from_raven_files(raven_file, None, "path1")
    assert ba.audio_files[0] == "path1"
    assert len(ba.audio_files) == 1
    ba = BoxedAnnotations.from_raven_files(Path(raven_file), None, Path("path1"))
    assert str(ba.audio_files[0]) == "path1"
    assert len(ba.audio_files) == 1


def test_to_and_from_crowsetta(boxed_annotations_2_files):
    # smoke test: BoxedAnnotations to crowsetta.Annotation list, and back

    # test 'bbox' mode:
    ba = boxed_annotations_2_files
    anns = ba.to_crowsetta()
    assert type(anns[0]) == crowsetta.Annotation
    assert type(anns[0].bboxes) == list
    assert type(anns[0].bboxes[0]) == crowsetta.BBox
    assert len(anns) == 2

    # back to BoxedAnnotations format
    ba2 = BoxedAnnotations.from_crowsetta(
        anns, audio_files=ba.audio_files, annotation_files=ba.annotation_files
    )

    # Note: order of annotations is not retained
    # because of the .groupby call
    assert set(ba2.df.annotation) == set([None, "a", "b"])

    # should contain .audio_files and .annotation_files
    assert ba2.annotation_files == ba.annotation_files
    assert ba2.audio_files == ba.audio_files

    # test 'sequence' mode:
    anns = ba.to_crowsetta(mode="sequence")
    assert type(anns[0]) == crowsetta.Annotation
    assert type(anns[0].seq) == crowsetta.Sequence

    # back to BoxedAnnotations format
    ba3 = BoxedAnnotations.from_crowsetta(
        anns, audio_files=ba.audio_files, annotation_files=ba.annotation_files
    )

    # should contain .audio_files and .annotation_files
    assert ba2.annotation_files == ba.annotation_files
    assert ba2.audio_files == ba.audio_files

    # order of annotations is not retained
    # because of the .groupby call
    assert set(ba3.df.annotation) == set([None, "a", "b"])


def test_crowsetta_annotation_id(boxed_annotations_2_files):
    # if annotation_id is in the dataframe columns, to_crowsetta
    # should create one Annotation per annotation_id for each
    # unique audio_file+annotation_file combo, rather than just one
    ba = boxed_annotations_2_files
    ba.df["annotation_id"] = [0, 1, 2]
    anns = ba.to_crowsetta(mode="bbox")
    assert type(anns[0]) == crowsetta.Annotation
    assert len(anns) == 3
    assert type(anns[0].bboxes[0]) == crowsetta.BBox

    # test with Sequence mode as well to be safe
    anns = ba.to_crowsetta(mode="sequence")
    assert type(anns[0]) == crowsetta.Annotation
    assert len(anns) == 3
    assert type(anns[0].seq) == crowsetta.Sequence

    # if user passes `ignore_sequence_id`, should create one Sequence
    anns = ba.to_crowsetta(mode="sequence", ignore_sequence_id=True)
    assert type(anns[0]) == crowsetta.Annotation
    assert type(anns[0].seq) == crowsetta.Sequence


def test_crowsetta_sequence_id(boxed_annotations_2_files):
    # if sequence_id is in the dataframe columns, to_crowsetta with
    # mode 'sequence' should create a _list_ of Sequences for each Annotation,
    # with one Sequence for each unique value of sequence_id
    ba = boxed_annotations_2_files
    ba.df["sequence_id"] = [0, 1, 2]
    anns = ba.to_crowsetta(mode="sequence")
    assert type(anns[0]) == crowsetta.Annotation
    assert type(anns[0].seq) == list
    assert type(anns[0].seq[0]) == crowsetta.Sequence

    # if user passes `ignore_sequence_id`, should create one Sequence
    anns = ba.to_crowsetta(mode="sequence", ignore_sequence_id=True)
    assert type(anns[0]) == crowsetta.Annotation
    assert type(anns[0].seq) == crowsetta.Sequence


def test_from_crowsetta_bbox():
    bbox = crowsetta.BBox(
        onset=0.0, offset=0.2, low_freq=0.0, high_freq=1000, label="a"
    )
    ba = BoxedAnnotations.from_crowsetta_bbox(bbox, "af", "anf")
    assert type(ba) == BoxedAnnotations
    assert len(ba.df) == 1
    assert set(ba.df["annotation"].values) == set(["a"])


def test_from_crowsetta_seq():
    seq = crowsetta.Sequence.from_dict(
        {
            "onsets_s": [0.0, 1.0],
            "offsets_s": [0.2, 1.2],
            "labels": ["a", "b"],
        }
    )
    ba = BoxedAnnotations.from_crowsetta_seq(seq, "af", "anf")
    assert type(ba) == BoxedAnnotations
    assert len(ba.df) == 2
    assert set(ba.df["annotation"].values) == set(["a", "b"])


def test_df_to_crowsetta_bbox(boxed_annotations):
    bboxes = annotations._df_to_crowsetta_bboxes(boxed_annotations.df)
    assert type(bboxes[0]) == crowsetta.BBox
    assert len(bboxes) == 3


def test_df_to_crowsetta_sequence(boxed_annotations):
    sequence = annotations._df_to_crowsetta_sequence(boxed_annotations.df)
    assert type(sequence) == crowsetta.Sequence
    assert len(sequence.onsets_s) == 3


def test_df_to_crowsetta_sequence(boxed_annotations):
    sequence = annotations._df_to_crowsetta_sequence(boxed_annotations.df)
    assert type(sequence) == crowsetta.Sequence
    assert len(sequence.onsets_s) == 3


def test_to_from_csv(boxed_annotations, saved_csv):
    # to csv
    boxed_annotations.to_csv(saved_csv)
    # from csv
    loaded = BoxedAnnotations.from_csv(saved_csv)
    assert type(loaded) == BoxedAnnotations

    # check for equality
    assert boxed_annotations.df.equals(loaded.df)


def test_find_overlapping_idxs_in_clip_df(boxed_annotations):
    clip_df = generate_clip_times_df(5, clip_duration=1.0, clip_overlap=0)
    # make it a multi-index, with the first level being the audio file, second being start, third being end time
    clip_df["audio_file"] = "audio_file.wav"
    clip_df = clip_df.set_index(["audio_file", "start_time", "end_time"])
    # annotation overlaps with 1 time-window
    idxs = annotations.find_overlapping_idxs_in_clip_df(
        "audio_file.wav", 0, 1, clip_df, min_label_overlap=0.25
    )
    assert len(idxs) == 1
    # annotation overlaps with 2 time-windows
    idxs = annotations.find_overlapping_idxs_in_clip_df(
        "audio_file.wav", 0, 1.3, clip_df, min_label_overlap=0.25
    )
    assert len(idxs) == 2
    # annotation-overlaps with no time-windows
    idxs = annotations.find_overlapping_idxs_in_clip_df(
        "audio_file.wav", 1000, 1001, clip_df, min_label_overlap=0.25
    )
    assert len(idxs) == 0


def test_categorical_labels_init(labels_df, labels_df_int):
    # label df with lists of string class labels
    classes = ["a", "b", "c"]
    cl = annotations.CategoricalLabels(
        files=labels_df["file"],
        start_times=labels_df["start_time"],
        end_times=labels_df["end_time"],
        labels=labels_df["labels"],
        classes=classes,
        integer_labels=False,
    )
    # classes may be in any order when inferred from labels
    assert cl.classes == classes
    # test @property labels and class_labels
    assert cl.class_labels == labels_df["labels"].to_list()
    assert cl.labels == labels_df_int["labels"].to_list()

    # label df with lists of integer class indices

    cl = annotations.CategoricalLabels(
        files=labels_df_int["file"],
        start_times=labels_df_int["start_time"],
        end_times=labels_df_int["end_time"],
        labels=labels_df_int["labels"],
        classes=classes,
        integer_labels=True,
    )
    assert cl.classes == classes
    # test @property labels and class_labels
    assert cl.class_labels == labels_df["labels"].to_list()
    assert cl.labels == labels_df_int["labels"].to_list()

    # test properties multihot_sparse and multihot_dense
    assert cl.multihot_dense.tolist() == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    assert cl.multihot_sparse.todense().tolist() == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]

    # test properties multihot_df_sparse, multihot_df_dense
    assert cl.multihot_df_sparse.values.tolist() == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    assert cl.multihot_df_dense.values.tolist() == [[1, 1, 0], [0, 1, 1], [1, 0, 1]]

    # test properties labels_at_index, multihot_labels_at_index
    assert cl.labels_at_index(0) == ["a", "b"]
    assert list(cl.multihot_labels_at_index(0)) == [1, 1, 0]


def test_categorical_labels_init_no_classes(labels_df):

    # init with classes=None
    cl = annotations.CategoricalLabels(
        files=labels_df["file"],
        start_times=labels_df["start_time"],
        end_times=labels_df["end_time"],
        labels=labels_df["labels"],
        classes=None,
        integer_labels=True,
    )
    assert set(cl.classes) == set(["a", "b", "c"])


def test_categorical_labels_from_categorical_labels_df(labels_df):
    # init with classes=None
    cl = annotations.CategoricalLabels.from_categorical_labels_df(
        labels_df, classes=None
    )
    assert set(cl.classes) == set(["a", "b", "c"])

    # init with class list
    cl = annotations.CategoricalLabels.from_categorical_labels_df(
        labels_df, classes=["a", "b", "c"]
    )
    assert cl.classes == ["a", "b", "c"]


def test_categorical_labels_from_multihot_df():
    # define multi-hot dataframe
    multi_hot_df = pd.DataFrame(
        {
            "file": ["f0", "f0", "f1"],
            "start_time": [0, 1, 0],
            "end_time": [1, 2, 1],
            "class0": [1, 0, 1],
            "class1": [0, 1, 1],
        },
    ).set_index(["file", "start_time", "end_time"])

    # convert to CategoricalLabels
    cl = annotations.CategoricalLabels.from_multihot_df(multi_hot_df)
    assert cl.classes == ["class0", "class1"]
    assert cl.labels == [[0], [1], [0, 1]]
    assert cl.class_labels == [["class0"], ["class1"], ["class0", "class1"]]


def test_train_test_split(boxed_annotations_2_files):
    train, test = boxed_annotations_2_files.train_test_split(
        train_size=0.5, random_state=0
    )
    assert len(train.audio_files) == 1
    assert len(test.audio_files) == 1
    assert train.audio_files[0] != test.audio_files[0]
    assert len(train.df) + len(test.df) == len(boxed_annotations_2_files.df)
    assert isinstance(train, BoxedAnnotations)
    assert isinstance(test, BoxedAnnotations)
