"""helpers for integrating with WandB and exporting content"""
import wandb
import pandas as pd
from opensoundscape.audio import Audio


def wandb_table(
    dataset,
    n=None,
    classes_to_extract=(),
    random_state=None,
    raise_exceptions=False,
    drop_labels=False,
):
    """Generate a wandb Table visualizing n random samples from a sample_df

    Args:
        dataset: object to generate samples, eg AudioFileDataset or AudioSplittingDataset
        n: number of samples to generate (randomly selected from df)
            - if None, does not subsample or change order
        bypass_augmentations: if True, augmentations in Preprocessor are skipped
        classes_to_extract: tuple of classes - will create columns containing the scores/labels
        random_state: default None; if integer provided, used for reproducible random sample
        drop_labels: if True, does not include 'label' column in Table

    Returns: a W&B Table of preprocessed samples with labels and playable audio

    """
    # select which clips to generate
    if n is None or len(dataset) < n:
        # if not enough samples to make n, just use all of them (don't complain)
        pass
    else:
        # randomly choose entire files from the table of labels
        dataset = dataset.sample(n=n, random_state=random_state)

    # set up columns for WandB display table
    table_columns = ["audio", "tensor", "labels", "path"]
    if dataset.has_clips:  # keep track of clip start/ends
        table_columns += ["clip start time"]
        table_columns += ["clip duration"]
    for c in classes_to_extract:
        table_columns += c

    sample_table = pd.DataFrame(columns=table_columns)
    for i in range(len(dataset)):
        try:
            # generate the content for a new row of the table
            sample = dataset[i]
            audio = Audio.from_file(
                sample.source, offset=sample.start_time, duration=sample.duration
            )

            row_info = [
                wandb.Audio(audio.samples, audio.sample_rate),  # audio object
                wandb.Image(sample.data * -1),  # spectrogram image
                sample.categorical_labels,
                str(sample.source),
            ]
            if dataset.has_clips:
                row_info += [sample.start_time, sample.duration]
            for c in classes_to_extract:
                row_info += [sample.labels[c]]
            # add new row to wandb Table
            sample_table.loc[len(sample_table)] = row_info

        except:  # by default, ignore exceptions
            if raise_exceptions:
                raise

    if drop_labels:
        sample_table = sample_table.drop(columns=["labels"])

    return wandb.Table(dataframe=sample_table)
