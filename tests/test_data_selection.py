#!/usr/bin/env python3
import pytest
import opensoundscape.data_selection as selection
import pandas as pd


def test_expand_multi_labeled_dataframe():
    input_df = pd.DataFrame({"Labels": ["hello|world"]})
    output_df = pd.DataFrame({"Labels": ["hello", "world"]})

    assert selection.expand_multi_labeled(input_df).equals(output_df)


def test_train_valid_binary_split():
    input_df = pd.DataFrame(
        {
            "Labels": [
                "hello",
                "world",
                "hello",
                "world",
                "hello",
                "world",
                "hello",
                "world",
                "hello",
                "world",
            ]
        }
    )

    train_df, valid_df = selection.binary_train_valid_split(input_df, "hello")

    assert train_df[train_df["Labels"] == "hello"]["Labels"].count() == 4
    assert valid_df[valid_df["Labels"] == "hello"]["Labels"].count() == 1
    assert train_df[train_df["Labels"] == "world"]["Labels"].count() == 4
    assert valid_df[valid_df["Labels"] == "world"]["Labels"].count() == 1
