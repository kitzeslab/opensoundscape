#!/usr/bin/env python3
import pandas as pd
from itertools import repeat


def resample(
    df, n_samples_per_class, upsample=True, downsample=True, random_state=None
):
    """resample a one-hot encoded label df for a target n_samples_per_class

    args:
        df: dataframe with one-hot encoded labels: columns are classes, index is sample name/path
        n_samples_per_class: target number of samples per class
        upsample: if True, duplicate samples for classes with <n samples to get to n samples
        downsample: if True, randomly sample classis with >n samples to get to n samples
        random_state: passed to np.random calls. If None, random state is not fixed.

    Note: The algorithm assumes that the label df is single-label.
    If the label df is multi-label, some classes can end up over-represented.

    Note 2: The resulting df will have samples ordered by class label, even if the input df
    had samples in a random order.
    """

    label_counts = df.sum(0)

    if min(label_counts) < 1 and upsample:
        raise ValueError("Cannot upsample when some classes have zero samples")

    class_dfs = [None] * len(df.columns)
    for idx, unique_label in enumerate(df.columns):
        sub_df = df[df[unique_label] == 1]
        n_class_samples = sub_df.shape[0]

        if n_class_samples < n_samples_per_class and (not upsample):
            # we don't want to upsample, so just keep these samples
            class_dfs[idx] = sub_df
            continue
        if n_class_samples > n_samples_per_class and (not downsample):
            # we don't want to downsample, so just keep all of samples
            class_dfs[idx] = sub_df
            continue

        # upsample or downsample as needed to get to n samples
        num_replicates, remainder = divmod(n_samples_per_class, n_class_samples)

        # take a random sample for the "remainder" portion
        # this is the entirety of the new set of n samples if downsampling,
        # and the samples with an 'extra' representation if upsampling
        random_df = sub_df.sample(n=remainder, random_state=random_state)

        # if upsampling, repeat all of the samples as many times as necessary
        if num_replicates > 0:
            repeat_df = pd.concat(repeat(sub_df, num_replicates))
            class_dfs[idx] = pd.concat([repeat_df, random_df])
        else:
            class_dfs[idx] = random_df

    return pd.concat(class_dfs)


def upsample(input_df, label_column="Labels", random_state=None):
    """Given a input DataFrame of categorical labels, upsample to maximum value

    Upsampling removes the class imbalance in your dataset. Rows for each label
    are repeated up to `max_count // rows`. Then, we randomly sample the rows
    to fill up to `max_count`.

    The input df is NOT one-hot encoded in this case, but instead contains
    categorical labels in a specified label_columns

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
