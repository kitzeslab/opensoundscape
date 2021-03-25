#!/usr/bin/env python3
import pytest
import opensoundscape.raven as raven
from pathlib import Path
import pandas as pd
import numpy as np
import numpy.testing as npt


@pytest.fixture()
def raven_warn_dir():
    return "./tests/raven_warn"


@pytest.fixture()
def raven_short_okay_dir():
    return "./tests/raven_okay_short"


@pytest.fixture()
def raven_long_okay_dir():
    return "./tests/raven_okay_long"


@pytest.fixture()
def raven_annotations_lower_okay_short(request, raven_short_okay_dir):
    raven.lowercase_annotations(raven_short_okay_dir)
    path = Path(f"{raven_short_okay_dir}/ShortExample.Table.1.selections.txt.lower")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def raven_annotations_lower_okay_long(request, raven_long_okay_dir):
    raven.lowercase_annotations(raven_long_okay_dir)
    path = Path(f"{raven_long_okay_dir}/LongExample.Table.1.selections.txt.lower")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def raven_annotations_lower_warn(request, raven_warn_dir):
    raven.lowercase_annotations(raven_warn_dir)
    path = Path(f"{raven_warn_dir}/Example.Table.1.selections.txt.lower")

    def fin():
        path.unlink()

    request.addfinalizer(fin)
    return path


def test_raven_annotation_check_on_okay(raven_short_okay_dir):
    raven.annotation_check(raven_short_okay_dir, col="class")


def test_raven_annotation_check_on_missing_col_warns(raven_short_okay_dir):
    with pytest.warns(UserWarning):
        raven.annotation_check(raven_short_okay_dir, col="col_that_doesnt_exist")


def test_raven_annotation_check_on_missing_label_warns(raven_warn_dir):
    with pytest.warns(UserWarning):
        raven.annotation_check(raven_warn_dir, col="class")


def test_raven_lowercase_annotations_on_okay(
    raven_short_okay_dir, raven_annotations_lower_okay_short
):
    assert raven_annotations_lower_okay_short.exists()


def test_raven_generate_class_corrections_with_okay(
    raven_short_okay_dir, raven_annotations_lower_okay_short
):
    csv = raven.generate_class_corrections(raven_short_okay_dir, col="class")
    assert csv == "raw,corrected\nhello,hello\n"


def test_raven_generate_class_corrections_with_empty_labels(
    raven_warn_dir, raven_annotations_lower_warn
):
    csv = raven.generate_class_corrections(raven_warn_dir, col="class")
    assert csv == "raw,corrected\nunknown,unknown\n"


def test_raven_generate_class_corrections_check_on_missing_col_warns(
    raven_warn_dir, raven_annotations_lower_warn, col="class"
):
    with pytest.warns(UserWarning):
        raven.generate_class_corrections(raven_warn_dir, col="col_that_doesnt_exist")


def test_raven_query_annotations_with_okay(
    raven_short_okay_dir, raven_annotations_lower_okay_short
):
    output = raven.query_annotations(raven_short_okay_dir, col="class", cls="hello")
    file_path = Path(raven_annotations_lower_okay_short)
    true_keys = [file_path]
    true_vals = pd.read_csv(file_path, sep="\t")
    assert list(output.keys()) == true_keys
    assert len(list(output.values())) == 1
    pd.testing.assert_frame_equal(list(output.values())[0], true_vals)


def test_raven_query_annotations_check_on_missing_col_warns(
    raven_short_okay_dir, raven_annotations_lower_okay_short
):
    with pytest.warns(UserWarning):
        raven.query_annotations(
            raven_short_okay_dir, cls="hello", col="col_that_doesnt_exist"
        )


def test_raven_split_single_annotation_short(raven_annotations_lower_okay_short):
    result_df = raven.split_single_annotation(
        raven_annotations_lower_okay_short, col="class", split_len_s=5
    )
    # print(result_df['seg_start'])
    print(result_df.hello.values)
    print(len(result_df.hello.values))
    npt.assert_array_equal(result_df.columns.values, ["seg_start", "seg_end", "hello"])
    npt.assert_array_equal(result_df["seg_start"].values, list(range(0, 381, 5)))
    npt.assert_array_equal(result_df["seg_end"].values, list(range(5, 386, 5)))
    npt.assert_array_equal(result_df["hello"].values, [*[0] * 71, *[1] * 6])


def test_raven_split_single_annotation_long_skiplast(raven_annotations_lower_okay_long):
    result_df = raven.split_single_annotation(
        raven_annotations_lower_okay_long, col="class", split_len_s=5
    )
    # print(result_df['seg_start'])
    print(len(result_df.woth.values))
    print(result_df.woth.values)
    print(result_df.eato.values)
    npt.assert_array_equal(
        result_df.columns.values, ["seg_start", "seg_end", "woth", "eato"]
    )
    npt.assert_array_equal(result_df["seg_start"].values, list(range(0, 26, 5)))
    npt.assert_array_equal(result_df["seg_end"].values, list(range(5, 31, 5)))
    npt.assert_array_equal(result_df["woth"].values, [1, 1, 1, 1, 1, 1])
    npt.assert_array_equal(result_df["eato"].values, [0, 1, 1, 1, 1, 1])


def test_raven_split_single_annotation_long_includelast(
    raven_annotations_lower_okay_long,
):
    result_df = raven.split_single_annotation(
        raven_annotations_lower_okay_long, col="class", split_len_s=5, keep_final=True
    )
    # print(result_df['seg_start'])
    print(len(result_df.woth.values))
    print(result_df.woth.values)
    print(result_df.eato.values)
    npt.assert_array_equal(
        result_df.columns.values, ["seg_start", "seg_end", "woth", "eato"]
    )
    npt.assert_array_equal(result_df["seg_start"].values, list(range(0, 31, 5)))
    npt.assert_array_equal(result_df["seg_end"].values, list(range(5, 36, 5)))
    npt.assert_array_equal(result_df["woth"].values, [1, 1, 1, 1, 1, 1, 0])
    npt.assert_array_equal(result_df["eato"].values, [0, 1, 1, 1, 1, 1, 1])
