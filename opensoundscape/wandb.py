"""helpers for integrating with WandB and exporting content"""
import wandb
from opensoundscape.torch.datasets import AudioFileDataset
import pandas as pd
from opensoundscape.annotations import one_hot_to_categorical
from opensoundscape.audio import Audio


def wandb_table(dataset, n, random_state=None):
    """Generate a wandb Table visualizing n random samples from a sample_df

    Args:
        dataset: object to generate samples, eg AudioFileDataset or AudioSplittingDataset
        n: number of samples to generate (randomly selected from df)
        bypass_augmentations: if True, augmentations in Preprocessor are skipped
        random_state: default None; if integer provided, used for reproducible random sample

    Returns: a W&B Table of preprocessed samples with labels and playable audio
    """
    # if not enough samples to make n, just use all of them (don't complain)
    if len(dataset) < n:
        n = len(dataset)

    # set up columns for WandB display table
    table_columns = ["audio", "tensor", "labels", "path"]

    # if the dataset specifies clip start/end times, randomly choose n clips
    # otherwise, randomly choose entire files from the table of labels
    if dataset.clip_times_df is not None:
        dataset = dataset.sample_clip_times_df(n=n, random_state=random_state)
        table_columns += ["clip start time"]
        table_columns += ["clip end time"]
    else:
        dataset = dataset.sample(n=n, random_state=random_state)
    classes = dataset.label_df.columns

    sample_table = pd.DataFrame(columns=table_columns)
    for i in range(len(dataset)):
        try:
            # generate the content for a new row of the table
            path = dataset.label_df.index.values[i]
            sample = dataset[i]
            if dataset.clip_times_df is not None:
                clip_row = dataset.clip_times_df.iloc[i]
                audio = Audio.from_file(
                    path,
                    offset=clip_row["start_time"],
                    duration=clip_row["end_time"] - clip_row["start_time"],
                )
            else:
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
            if dataset.clip_times_df is not None:
                row_info += [
                    clip_row["start_time"],
                    clip_row["end_time"],
                ]

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


#     scores, _, _ = model.predict(sample_df)

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
