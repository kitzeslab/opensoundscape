#!/usr/bin/env python3
import pandas as pd
from sklearn.model_selection import train_test_split
from copy import copy
from itertools import repeat, count


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


def train_valid_split(
    input_df, label_column="Labels", train_size=0.8, random_state=101
):
    """ Split a dataframe into train and validation dataframes

    Given an input dataframe with a labels column split each unique label into
    a train size and 1 - train_size for training and validation sets. If
    label_column is `None` don't stratify.

    Args:
        input_df:       A dataframe
        label_column:   Name of the column that labels should come from [default: "Labels"]
        train_size:     The decimal fraction to use for the training set [default: 0.8]
        random_state:   The random state to use for train_test_split [default: 101]

    Output:
        train_df:       A Dataframe containing the training set
        valid_df:       A Dataframe containing the validation set
    """

    df = copy(input_df)

    if label_column:
        assert label_column in df.columns
        all_labels = df[label_column].unique()

        train_dfs = [None] * len(all_labels)
        valid_dfs = [None] * len(all_labels)
        for idx, label in enumerate(all_labels):
            selection = df[df[label_column] == label]
            train, valid = train_test_split(
                selection, train_size=train_size, random_state=random_state
            )
            train_dfs[idx] = train
            valid_dfs[idx] = valid
    else:
        train, valid = train_test_split(
            df, train_size=train_size, random_state=random_state
        )
        train_dfs = [train]
        valid_dfs = [valid]

    return pd.concat(train_dfs), pd.concat(valid_dfs)


def add_binary_numeric_labels(
    input_df, label, input_column="Labels", output_column="NumericLabels"
):
    """ Add binary numeric labels to dataframe based on label

    Given a dataframe and a label from input_column produce a new
    dataframe with an output_column and a label map

    Args:
        input_df:       A dataframe
        label:          The label to set to 1
        input_column:   The column to read labels from
        output_column:  The column to write numeric labels to

    Output:
        output_df:      A dataframe with an additional output_column
        label_map:      A dictionary, keys are f"not_{label}" and f"{label}", values are 0 and 1
    """

    df = copy(input_df)
    df[output_column] = df[input_column].apply(lambda x: 1 if x == label else 0)
    label_map = {f"not_{label}": 0, f"{label}": 1}
    return df, label_map


def add_numeric_labels(input_df, input_column="Labels", output_column="NumericLabels"):
    """ Add numeric labels to dataframe

    Given a dataframe with input_column produce a new dataframe with an
    output_column and a label map

    Args:
        input_df:       A dataframe
        input_column:   The column to read labels from
        output_column:  The column to write numeric labels to

    Output:
        output_df:      A dataframe with an additional output_column
        label_map:      A dictionary, keys are the unique labels and monotonically increasing values starting at 0
    """

    df = copy(input_df)
    unique_labels = df[input_column].unique()
    label_map = {k: v for k, v in zip(unique_labels, count(0))}
    df[output_column] = df[input_column].apply(lambda x: label_map[x])
    return df, label_map


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
