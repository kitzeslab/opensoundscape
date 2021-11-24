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
    for _ in range(5):
        resample_df = selection.resample(resample_df, 2)
        assert resample_df.shape[0] == 6


def test_resample_no_upsample(resample_df):
    for _ in range(5):
        resample_df = selection.resample(resample_df, 2, upsample=False)
        assert resample_df.shape[0] == 5


def test_resample_no_downsample(resample_df):
    for _ in range(5):
        resample_df = selection.resample(resample_df, 2, downsample=False)
        assert resample_df.shape[0] == 8


def test_upsample_basic(upsample_df):
    for _ in range(100):
        upsampled_df = selection.upsample(upsample_df)
        assert upsampled_df.shape[0] == 20
