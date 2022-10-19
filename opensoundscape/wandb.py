"""helpers for integrating with WandB and exporting content"""
import wandb
from opensoundscape.torch.datasets import AudioFileDataset
import pandas as pd
from opensoundscape.annotations import one_hot_to_categorical


def wandb_table(
    sample_df, n, preprocessor, bypass_augmentations=False, random_state=None
):
    """Generate a wandb Table visualizing n random samples from a sample_df

    Args:
        sample_df: one-hot label df (0/1) with paths to samples as the index, classes as columns
        n: number of samples to generate (randomly selected from df)
        preprocessor: preprocessor object used to generate samples
        bypass_augmentations: if True, augmentations in Preprocessor are skipped
        random_state: default None; if integer provided, used for reproducible random sample

    Returns: a W&B Table of preprocessed samples with labels and playable audio
    """
    samples = sample_df.sample(n, random_state=random_state)
    inspection_dataset = AudioFileDataset(samples, preprocessor)
    inspection_dataset.bypass_augmentations = bypass_augmentations
    classes = sample_df.columns

    sample_table = pd.DataFrame(columns=["audio", "tensor", "labels", "path"])
    for i in range(len(inspection_dataset)):
        try:
            sample = inspection_dataset[i]
            path = samples.index.values[i]
            sample_table.loc[len(sample_table)] = [
                wandb.Audio(str(path)),
                wandb.Image(sample["X"] * -1),
                one_hot_to_categorical([sample["y"]], classes)[0],
                str(path),
            ]
        except:
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
