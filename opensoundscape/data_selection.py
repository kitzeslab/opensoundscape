#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import copy
from itertools import repeat


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


def binary_train_valid_split(
    input_df, label, label_column="Labels", train_size=0.8, random_state=101
):
    """ Split a dataset into train and validation dataframes

    Given a Dataframe and a label in column "Labels" (singly labeled) generate
    a train dataset with ~80% of each label and a valid dataset with the rest.

    Args:
        input_df:       A singly-labeled CSV file
        label:          One of the labels in the column label_column to use as a positive
                            label (1), all others are negative (0)
        label_column:   Name of the column that labels should come from [default: "Labels"]
        train_size:     The decimal fraction to use for the training set [default: 0.8]
        random_state:   The random state to use for train_test_split [default: 101]

    Output:
        train_df:       A Dataframe containing the training set
        valid_df:       A Dataframe containing the validation set
    """

    assert label_column in input_df.columns
    df = copy(input_df)

    all_labels = df[label_column].unique()
    df["NumericLabels"] = df[label_column].apply(lambda x: 1 if x == label else 0)

    train_dfs = [None] * len(all_labels)
    valid_dfs = [None] * len(all_labels)
    for idx, label in enumerate(all_labels):
        selection = df[df[label_column] == label]
        train, valid = train_test_split(
            selection, train_size=train_size, random_state=random_state
        )
        train_dfs[idx] = train
        valid_dfs[idx] = valid

    return pd.concat(train_dfs), pd.concat(valid_dfs)


def upsample(input_df, label_column="Labels", random_state=None):
    """ Given a input DataFrame upsample to maximum value

    Upsampling removes the class imbalance in your dataset. Rows for each label
    are repeated up to `max_count // rows`. Then, we randomly sample the rows
    to fill up to `max_count`.

    Input:
        input_df:       A DataFrame to upsample
        label_column:   The column to draw unique labels from
        random_state:   Set the random_state during sampling

    Output:
        df:             An upsampled DataFrame
    """

    unique_labels = input_df[label_column].unique()
    label_counts = input_df.groupby(by=label_column)[label_column].count()
    max_count = label_counts.max()

    dfs = [None] * unique_labels.shape[0]
    for idx, unique_label in enumerate(unique_labels):
        sub_df = input_df[input_df[label_column] == unique_label]
        num_replicates, remainder = divmod(max_count, sub_df.shape[0])

        if random_state:
            random_df = sub_df.sample(n=remainder, random_state=random_state)
        else:
            random_df = sub_df.sample(n=remainder)

        repeat_df = pd.concat(repeat(sub_df, num_replicates))

        dfs[idx] = pd.concat([repeat_df, random_df])

    return pd.concat(dfs)
