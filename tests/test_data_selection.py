#!/usr/bin/env python3
import pytest
import opensoundscape.data_selection as selection
import pandas as pd


@pytest.fixture
def upsample_df():
    return pd.read_csv("tests/to_upsample.csv")


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


def test_expand_multi_labeled_dataframe():
    input_df = pd.DataFrame({"Labels": ["foo|bar"]})
    output_df = pd.DataFrame({"Labels": ["foo", "bar"]})

    assert selection.expand_multi_labeled(input_df).equals(output_df)


def test_train_valid_basic_split(input_dataframe):
    train_df, valid_df = selection.train_valid_split(input_dataframe)

    assert train_df[train_df["Labels"] == "foo"]["Labels"].count() == 4
    assert valid_df[valid_df["Labels"] == "foo"]["Labels"].count() == 1
    assert train_df[train_df["Labels"] == "bar"]["Labels"].count() == 4
    assert valid_df[valid_df["Labels"] == "bar"]["Labels"].count() == 1
    assert train_df[train_df["Labels"] == "baz"]["Labels"].count() == 4
    assert valid_df[valid_df["Labels"] == "baz"]["Labels"].count() == 1


def test_add_binary_numeric_labels(input_dataframe):
    output_column = "test_label"
    output_df, label_map = selection.add_binary_numeric_labels(
        input_dataframe, "foo", output_column=output_column
    )
    assert len(label_map) == 2
    assert sorted(output_df[output_column].unique()) == [0, 1]


def test_add_numeric_labels(input_dataframe):
    output_column = "test_label"
    output_df, label_map = selection.add_numeric_labels(
        input_dataframe, output_column=output_column
    )
    assert len(label_map) == 3
    assert sorted(output_df[output_column].unique()) == [0, 1, 2]


def test_upsample_basic(upsample_df):
    for _ in range(100):
        upsampled_df = selection.upsample(upsample_df)
        assert upsampled_df.shape[0] == 20
