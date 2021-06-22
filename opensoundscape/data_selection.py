#!/usr/bin/env python3
import pandas as pd
from itertools import repeat


def upsample(input_df, label_column="Labels", random_state=None):
    """Given a input DataFrame upsample to maximum value

    Upsampling removes the class imbalance in your dataset. Rows for each label
    are repeated up to `max_count // rows`. Then, we randomly sample the rows
    to fill up to `max_count`.

    Args:
        input_df:       A DataFrame to upsample
        label_column:   The column to draw unique labels from
        random_state:   Set the random_state during sampling

    Returns:
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
