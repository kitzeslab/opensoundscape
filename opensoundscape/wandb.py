"""helpers for integrating with WandB and exporting content"""
import wandb
import pandas as pd
from opensoundscape.annotations import one_hot_to_categorical
from opensoundscape.audio import Audio


def wandb_table(dataset, n=None, classes_to_extract=(), random_state=None):
    """Generate a wandb Table visualizing n random samples from a sample_df

    Args:
        dataset: object to generate samples, eg AudioFileDataset or AudioSplittingDataset
        n: number of samples to generate (randomly selected from df)
            - if None, does not subsample or change order
        bypass_augmentations: if True, augmentations in Preprocessor are skipped
        classes_to_extract: tuple of classes - will create columns containing the scores/labels
        random_state: default None; if integer provided, used for reproducible random sample

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
        table_columns += ["clip end time"]
    for c in classes_to_extract:
        table_columns += c

    classes = dataset.label_df.columns

    sample_table = pd.DataFrame(columns=table_columns)
    for i in range(len(dataset)):
        try:
            # generate the content for a new row of the table
            sample = dataset[i]
            clip_row = dataset.label_df.iloc[i]
            if dataset.has_clips:
                clip_row = dataset.label_df.iloc[i]
                path = clip_row.name[0]
                start_time = clip_row.name[1]
                end_time = clip_row.name[2]
                duration = end_time - start_time
                audio = Audio.from_file(
                    path,
                    offset=start_time,
                    duration=duration,
                )
            else:
                path = clip_row.name
                audio = Audio.from_file(path)

            row_info = [
                wandb.Audio(audio.samples, audio.sample_rate),
                wandb.Image(sample["X"] * -1),
                # list of "1" labels
                one_hot_to_categorical([sample["y"]], classes)[0]
                if "y" in sample
                else None,
                str(path),
            ]
            if dataset.has_clips:
                row_info += [start_time, end_time]
            for c in classes_to_extract:
                row_info += [clip_row[c]]
            # add new row to wandb Table
            sample_table.loc[len(sample_table)] = row_info

        except:  # we'll allow failures to pass here
            pass  # print(f"failed to load sample {sample_df.index.values[i]}")

    return wandb.Table(dataframe=sample_table)


# def wandb_prediction_table(sample_df, n, model, random_state=None):
#     """generate table to visualize samples alongside model prediction labels"""
#     samples = sample_df.sample(n, random_state=random_state)
#     inspection_dataset = AudioFileDataset(samples, model.preprocessor)
#     inspection_dataset.bypass_augmentations = True


#     scores = model.predict(sample_df)

#     sample_table = pd.DataFrame(columns=["audio", "tensor", "labels:scores", "path"])
#     for i in range(len(inspection_dataset)):
#         try:
#             sample = inspection_dataset[i]
#             path = samples.index.values[i]
#             labels = one_hot_to_categorical([sample["y"]], model.classes)[0]
#             sample_table.loc[len(sample_table)] = [
#                 wandb.Audio(path),
#                 wandb.Image(sample["X"] * -1),
#                 str(scores.loc[path][labels]),
#                 path,
#             ]
#         except:
#             pass  # print(f"failed to load sample {sample_df.index.values[i]}")
#     return wandb.Table(dataframe=sample_table)
