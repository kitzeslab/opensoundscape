import pytest
import pandas as pd
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.datasets import AudioFileDataset, AudioSplittingDataset

from opensoundscape.logging import wandb_table


@pytest.fixture()
def dataset_df():
    paths = ["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"]
    labels = [[1, 0], [0, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def pre():
    return SpectrogramPreprocessor(sample_duration=2.0)


def test_wandb_table_files(dataset_df, pre):
    """smoke test: does it make a wandb table?"""
    dataset = AudioFileDataset(dataset_df, pre)
    wandb_table(dataset, n=2, raise_exceptions=True)


def test_wandb_table_clips(dataset_df, pre):
    """smoke test: does it make a wandb table?"""
    dataset = AudioSplittingDataset(dataset_df, pre)
    wandb_table(dataset, n=2, raise_exceptions=True)


def test_wandb_table_no_labels(dataset_df, pre):
    """smoke test: does it make a wandb table?"""
    dataset = AudioSplittingDataset(dataset_df, pre)
    wandb_table(dataset, n=2, raise_exceptions=True, drop_labels=False)
