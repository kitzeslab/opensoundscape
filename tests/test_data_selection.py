#!/usr/bin/env python3
import pytest
import opensoundscape.data_selection as selection
import pandas as pd


@pytest.fixture
def upsample_df():
    return pd.read_csv("tests/csvs/to_upsample.csv")


@pytest.fixture
def resample_df():
    return pd.read_csv("tests/csvs/to_resample.csv").set_index("file")


@pytest.fixture
def multihot_multiindex_df():
    files = "aabbccccccccccccccc"
    files = list(files)
    return pd.DataFrame(
        {
            "file": files,
            "start_time": list(range(len(files))),
            "end_time": list(range(1, len(files) + 1)),
            "class1": [1] * len(files),
            "class2": [1] * len(files),
        }
    ).set_index(["file", "start_time", "end_time"])


@pytest.fixture
def multihot_multiindex_df_with_duplicates():
    files = "aabbccccccccccccccc"
    files = list(files)
    return pd.DataFrame(
        {
            "file": files,
            "start_time": [0] * len(files),
            "end_time": [2] * len(files),
            "class1": [1] * len(files),
            "class2": [1] * len(files),
        }
    ).set_index(["file", "start_time", "end_time"])


@pytest.fixture
def input_dataframe():
    return pd.DataFrame(
        {
            "Labels": [
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
                "foo",
                "bar",
                "baz",
            ]
        }
    )


def test_resample_basic(resample_df):
    # changed behavior: now retains all-0 rows by default
    for _ in range(5):
        df = selection.resample(resample_df, 2)
        assert df.shape[0] == 2 * 3 + 2


def test_resample_no_upsample(resample_df):
    for _ in range(5):
        df = selection.resample(resample_df, 2, upsample=False)
        assert df.shape[0] == 5 + 2


def test_resample_no_downsample(resample_df):
    for _ in range(5):
        df = selection.resample(resample_df, 2, downsample=False)
        assert df.shape[0] == 8 + 2


def test_resample_inclue_negatives(resample_df):
    for negatives in (0, 1, 5):
        df = selection.resample(
            resample_df, 1, n_samples_without_labels=negatives, downsample=True
        )
        assert df.shape[0] == 3 + negatives


def test_upsample_basic(upsample_df):
    for _ in range(100):
        upsampled_df = selection.upsample(upsample_df)
        assert upsampled_df.shape[0] == 20


def test_resample_no_negatives(resample_df):
    for _ in range(5):
        df = selection.resample(resample_df, 2, n_samples_without_labels=0)
        assert df.shape[0] == 2 * 3


def test_train_test_split(multihot_multiindex_df):
    # label df with (file,start_time,end_time) multi-index

    # split by file
    train_df, test_df = selection.train_test_split(
        multihot_multiindex_df, test_size=0.5, random_state=42, by_file=True
    )
    # make sure all rows with the same file name are in the same set
    train_files = set(train_df.index.get_level_values(0))
    test_files = set(test_df.index.get_level_values(0))
    assert train_files.isdisjoint(test_files)

    # split by row
    train_df, test_df = selection.train_test_split(
        multihot_multiindex_df, test_size=0.5, random_state=42, by_file=False
    )
    # make sure rows with the same file name can be in different sets
    trial = 0
    while trial < 50:
        train_df, test_df = selection.train_test_split(
            multihot_multiindex_df, test_size=0.5, random_state=trial, by_file=False
        )
        train_files = set(train_df.index.get_level_values(0))
        test_files = set(test_df.index.get_level_values(0))
        if not train_files.isdisjoint(test_files):
            break  # found a trial where files are in both sets
        trial += 1


def test_warns_data_leakage(multihot_multiindex_df_with_duplicates):
    with pytest.warns(UserWarning, match="duplicate"):
        selection.train_test_split(
            multihot_multiindex_df_with_duplicates,
            test_size=0.5,
            random_state=42,
            by_file=False,
        )
