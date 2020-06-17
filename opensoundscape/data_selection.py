#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import copy


def expand_multi_labeled(input_df):
    """ Given a multi-labeled dataframe, generate a singly-labeled dataframe

    Given a Dataframe with a "Labels" column that is multi-labeled (e.g. "hello|world")
    split the row into singly labeled rows.

    Args:
        input_df: A Dataframe with a multi-labeled "Labels" column (separated by "|")

    Output:
        output_df: A Dataframe with singly-labeled "Labels" column
    """
    assert "Labels" in input_df.columns

    df = copy(input_df)

    df["Labels"] = df["Labels"].str.split("|")

    return df.explode("Labels").reset_index(drop=True)


def binary_train_valid_split(input_df, label, train_size=0.8, random_state=101):
    """ Split a dataset into train and validation dataframes

    Given a Dataframe and a label in column "Labels" (singly labeled) generate
    a train dataset with ~80% of each label and a valid dataset with the rest.

    Args:
        input_df:       A singly-labeled CSV file with a column "Labels"
        label:          One of the labels in the column "Labels" to use as a positive
                            label (1), all others are negative (0)
        train_size:     The decimal fraction to use for the training set [default: 0.8]
        random_state:   The random state to use for train_test_split [default: 101]

    Output:
        train_df:       A Dataframe containing the training set
        valid_df:       A Dataframe containing the validation set
    """

    assert "Labels" in input_df.columns
    df = copy(input_df)

    all_labels = df["Labels"].unique()
    df["NumericLabels"] = df["Labels"].apply(lambda x: 1 if x == label else 0)

    train_dfs = [None] * len(all_labels)
    valid_dfs = [None] * len(all_labels)
    for idx, label in enumerate(all_labels):
        selection = df[df["Labels"] == label]
        train, valid = train_test_split(
            selection, train_size=train_size, random_state=random_state
        )
        train_dfs[idx] = train
        valid_dfs[idx] = valid

    return pd.concat(train_dfs), pd.concat(valid_dfs)
