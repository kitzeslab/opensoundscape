#!/usr/bin/env python3
import pytest
import opensoundscape.raven as raven
from pathlib import Path
import pandas as pd


@pytest.fixture()
def raven_warn_dir():
    return "./tests/raven_warn"


@pytest.fixture()
def raven_okay_dir():
    return "./tests/raven_okay"


@pytest.fixture()
def raven_annotations_lower_okay(request, raven_okay_dir):
    raven.lowercase_annotations(raven_okay_dir)
    path = Path(f"{raven_okay_dir}/Example.Table.1.selections.txt.lower")

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


def test_raven_annotation_check_on_okay(raven_okay_dir):
    raven.annotation_check(raven_okay_dir)


def test_raven_annotation_check_on_missing_col_warns(raven_okay_dir):
    with pytest.warns(UserWarning):
        raven.annotation_check(raven_okay_dir, col="col_that_doesnt_exist")


def test_raven_annotation_check_on_missing_label_warns(raven_warn_dir):
    with pytest.warns(UserWarning):
        raven.annotation_check(raven_warn_dir)


def test_raven_lowercase_annotations_on_okay(
    raven_okay_dir, raven_annotations_lower_okay
):
    assert raven_annotations_lower_okay.exists()


def test_raven_generate_class_corrections_with_okay(
    raven_okay_dir, raven_annotations_lower_okay
):
    csv = raven.generate_class_corrections(raven_okay_dir)
    assert csv == "raw,corrected\nhello,hello\n"


def test_raven_generate_class_corrections_with_empty_labels(
    raven_warn_dir, raven_annotations_lower_warn
):
    csv = raven.generate_class_corrections(raven_warn_dir)
    assert csv == "raw,corrected\nunknown,unknown\n"


def test_raven_generate_class_corrections_check_on_missing_col_warns(
    raven_warn_dir, raven_annotations_lower_warn
):
    with pytest.warns(UserWarning):
        raven.generate_class_corrections(raven_warn_dir, col="col_that_doesnt_exist")


def test_raven_query_annotations_with_okay(
    raven_okay_dir, raven_annotations_lower_okay
):
    output = raven.query_annotations(raven_okay_dir, cls="hello")
    file_path = Path(raven_annotations_lower_okay)
    true_keys = [file_path]
    true_vals = pd.read_csv(file_path, sep="\t")
    assert list(output.keys()) == true_keys
    assert len(list(output.values())) == 1
    pd.testing.assert_frame_equal(list(output.values())[0], true_vals)


def test_raven_query_annotations_check_on_missing_col_warns(
    raven_okay_dir, raven_annotations_lower_okay
):
    with pytest.warns(UserWarning):
        raven.query_annotations(
            raven_okay_dir, cls="hello", col="col_that_doesnt_exist"
        )
