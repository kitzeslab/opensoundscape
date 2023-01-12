#!/usr/bin/env python3
import pytest
from opensoundscape import annotations
from opensoundscape.annotations import BoxedAnnotations
from pathlib import Path
import pandas as pd
import numpy as np
from opensoundscape.helpers import generate_clip_times_df


@pytest.fixture()
def raven_file():
    return "tests/raven_annots/MSD-0003_20180427_2minstart00.Table.1.selections.txt"


@pytest.fixture()
def raven_file_empty():
    return "tests/raven_annots/EmptyExample.Table.1.selections.txt"


@pytest.fixture()
def saved_raven_file(request):
    path = Path("tests/raven_annots/saved_raven_file.txt")

    # remove this after tests are complete
    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def boxed_annotations():
    df = pd.DataFrame(
        data={
            "start_time": [0, 3, 4],
            "end_time": [1, 5, 5],
            "low_f": [0, 500, 1000],
            "high_f": [100, 1200, 1500],
            "annotation": ["a", "b", None],
        }
    )
    return BoxedAnnotations(df, audio_file=None)


def test_load_raven_annotations(raven_file):
    a = BoxedAnnotations.from_raven_file(raven_file, annotation_column="Species")
    assert len(a.df) == 10
    assert set(a.df["annotation"]) == {"WOTH", "EATO", "LOWA", np.nan}


def test_load_raven_no_annotation_column(raven_file):
    a = BoxedAnnotations.from_raven_file(raven_file, annotation_column=None)
    # we should now have a dataframe with a column "Species"
    assert len(a.df) == 10
    assert set(a.df["Species"]) == {"WOTH", "EATO", "LOWA", np.nan}


def test_load_raven_annotations_empty(raven_file_empty):
    a = BoxedAnnotations.from_raven_file(raven_file_empty, annotation_column="Species")
    assert len(a.df) == 0


def test_to_raven_file(boxed_annotations, saved_raven_file):
    assert not saved_raven_file.exists()
    boxed_annotations.to_raven_file(saved_raven_file)
    assert saved_raven_file.exists()


def test_subset(boxed_annotations):
    assert len(boxed_annotations.subset(["a", None]).df) == 2


def test_subset_to_nan(raven_file):
    a = BoxedAnnotations.from_raven_file(raven_file, annotation_column="Species")
    assert len(a.subset([np.nan]).df) == 1


def test_trim(boxed_annotations):
    trimmed = boxed_annotations.trim(0.5, 3.5, edge_mode="trim")
    assert len(trimmed.df) == 2
    assert np.max(trimmed.df["end_time"]) == 3.0
    assert np.min(trimmed.df["start_time"]) == 0.0


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


def test_global_one_hot_labels(boxed_annotations):
    assert boxed_annotations.global_one_hot_labels(classes=["a", "b", "c"]) == [1, 1, 0]


def test_one_hot_labels_like(boxed_annotations):
    clip_df = generate_clip_times_df(5, clip_duration=1.0, clip_overlap=0)
    labels = boxed_annotations.one_hot_labels_like(
        clip_df, class_subset=["a"], min_label_overlap=0.25
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())


def test_one_hot_labels_like_overlap(boxed_annotations):
    clip_df = generate_clip_times_df(3, clip_duration=1.0, clip_overlap=0.5)
    labels = boxed_annotations.one_hot_labels_like(
        clip_df, class_subset=["a"], min_label_overlap=0.25
    )
    assert np.array_equal(labels.values, np.array([[1, 1, 0, 0, 0]]).transpose())


def test_one_hot_clip_labels(boxed_annotations):
    labels = boxed_annotations.one_hot_clip_labels(
        full_duration=5,
        clip_duration=1.0,
        clip_overlap=0,
        class_subset=["a"],
        min_label_overlap=0.25,
    )
    assert np.array_equal(labels.values, np.array([[1, 0, 0, 0, 0]]).transpose())


def test_one_hot_clip_labels_overlap(boxed_annotations):
    labels = boxed_annotations.one_hot_clip_labels(
        full_duration=3,
        clip_duration=1.0,
        clip_overlap=0.5,
        class_subset=["a"],
        min_label_overlap=0.25,
    )
    assert np.array_equal(labels.values, np.array([[1, 1, 0, 0, 0]]).transpose())


def test_convert_labels(boxed_annotations):
    boxed_annotations = boxed_annotations.convert_labels({"a": "c"})
    assert set(boxed_annotations.df["annotation"]) == {"b", "c", None}


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


def test_one_hot_labels_on_time_interval(boxed_annotations):
    a = annotations.one_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0,
        end_time=3.5,
        min_label_overlap=0.25,
        classes=["a", "b"],
    )
    assert a["a"] == 1 and a["b"] == 1

    a = annotations.one_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0,
        end_time=3.5,
        min_label_overlap=0.75,
        classes=["a", "b"],
    )
    assert a["a"] == 1 and a["b"] == 0


def test_one_hot_labels_on_time_interval_fractional(boxed_annotations):
    """test min_label_fraction use cases"""
    # too short but satisfies fraction
    a = annotations.one_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0.4,
        end_time=3,
        min_label_overlap=2,
        min_label_fraction=0.5,
        classes=["a"],
    )
    assert a["a"] == 1

    # too short and not enough for fraction
    a = annotations.one_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0.4,
        end_time=3,
        min_label_overlap=2,
        min_label_fraction=0.9,
        classes=["a"],
    )
    assert a["a"] == 0

    # long enough, although less than fraction
    a = annotations.one_hot_labels_on_time_interval(
        boxed_annotations.df,
        start_time=0.4,
        end_time=3,
        min_label_overlap=0.5,
        min_label_fraction=0.9,
        classes=["a"],
    )
    assert a["a"] == 1


def test_categorical_to_one_hot():
    cat_labels = [["a", "b"], ["a", "c"]]
    one_hot, classes = annotations.categorical_to_one_hot(
        cat_labels, classes=["a", "b", "c", "d"]
    )
    assert set(classes) == {"a", "b", "c", "d"}
    assert np.array_equal(one_hot, [[1, 1, 0, 0], [1, 0, 1, 0]])

    # without passing classes list:
    one_hot, classes = annotations.categorical_to_one_hot(cat_labels)
    assert set(classes) == {"a", "b", "c"}


def test_one_hot_to_categorical():
    classes = ["a", "b", "c"]
    one_hot = [[0, 0, 1], [1, 1, 1]]
    cat_labels = annotations.one_hot_to_categorical(one_hot, classes)
    assert np.array_equal(cat_labels, [["c"], ["a", "b", "c"]])


def test_one_hot_to_categorical_and_back():
    classes = ["a", "b", "c"]
    one_hot = [[0, 0, 1], [1, 1, 1]]
    cat_labels = annotations.one_hot_to_categorical(one_hot, classes)
    one_hot2, classes2 = annotations.categorical_to_one_hot(cat_labels, classes)

    assert np.array_equal(one_hot, one_hot2)
    assert np.array_equal(classes, classes2)


# test robustness of raven methods for empty annotation file
def test_raven_annotation_methods_empty(raven_file_empty):
    a = BoxedAnnotations.from_raven_file(raven_file_empty, annotation_column="Species")

    a.trim(0, 5)
    a.bandpass(0, 11025)
    assert len(a.df) == 0

    # test with random parameters to generate clip dataframe
    clip_df = generate_clip_times_df(
        full_duration=10,
        clip_duration=2,
    )

    # classes = None
    labels_df = a.one_hot_labels_like(
        clip_df,
        classes=None,
        min_label_overlap=0.25,
    )

    assert (labels_df.reset_index() == clip_df).all().all()

    # classes = subset
    labels_df = a.one_hot_labels_like(
        clip_df,
        classes=["Species1", "Species2"],
        min_label_overlap=0.25,
    )

    assert len(labels_df) == len(clip_df)
    assert (labels_df.columns == ["Species1", "Species2"]).all()
