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
