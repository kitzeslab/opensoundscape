#!/usr/bin/env python3
import pytest
from pathlib import Path
from opensoundscape.datasets import SplitterDataset, SingleTargetAudioDataset
from torch.utils.data import DataLoader
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal


tmp_path = "tests/_tmp_split"


@pytest.fixture()
def temporary_split_storage(request):
    path = Path(tmp_path)
    path.mkdir()
    yield path
    path.rmdir()


@pytest.fixture()
def splitter_results_default(request):
    split0 = Path(f"{tmp_path}/a1f220f1640c2e35de106c930e5cd20f.wav")
    split1 = Path(f"{tmp_path}/f7ed400e5093fbafc09df578ff9c5425.wav")
    yield split0, split1
    split0.unlink()
    split1.unlink()


@pytest.fixture()
def splitter_results_last(request):
    split0 = Path(f"{tmp_path}/36fec9935a0d1c38da05004e5d096f6d.wav")
    split1 = Path(f"{tmp_path}/200c78d44bf9b3985d464e06f7cb395b.wav")
    yield split0, split1
    split0.unlink()
    split1.unlink()


@pytest.fixture()
def one_min_audio_list():
    return [Path("tests/audio/1min.wav")]


@pytest.fixture()
def single_target_audio_dataset_df():
    return pd.read_csv("tests/csvs/input.csv")


@pytest.fixture()
def single_target_audio_dataset_long_audio_df():
    return pd.DataFrame(
        {"Destination": ["tests/audio/great_plains_toad.wav"], "NumericLabels": [1]}
    )


def test_basic_splitting_operation_default(
    temporary_split_storage, splitter_results_default, one_min_audio_list
):
    dataset = SplitterDataset(
        one_min_audio_list,
        duration=25,
        overlap=0,
        output_directory=temporary_split_storage,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=SplitterDataset.collate_fn,
    )

    results = []
    for data in dataloader:
        for output in data:
            results.append(output)
    assert len(results) == 2

    split0, split1 = splitter_results_default
    assert split0.exists()
    assert split1.exists()


def test_basic_splitting_operation_with_include_last_segment(
    temporary_split_storage, splitter_results_last, one_min_audio_list
):
    dataset = SplitterDataset(
        one_min_audio_list,
        duration=30,
        overlap=0,
        output_directory=temporary_split_storage,
        include_last_segment=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=SplitterDataset.collate_fn,
    )

    results = []
    for data in dataloader:
        for output in data:
            results.append(output)
    assert len(results) == 2

    split0, split1 = splitter_results_last
    assert split0.exists()
    assert split1.exists()


def test_single_target_audio_dataset_default(single_target_audio_dataset_df):
    dataset = SingleTargetAudioDataset(
        single_target_audio_dataset_df,
        label_dict={0: "absent", 1: "present"},
        label_column="NumericLabels",
        height=225,
        width=226,
    )
    assert dataset[0]["X"].shape == (3, 225, 226)
    assert dataset[0]["y"].shape == (1,)


def test_single_target_audio_dataset_to_image(single_target_audio_dataset_df):
    dataset = SingleTargetAudioDataset(single_target_audio_dataset_df, label_dict=None)

    pixels = dataset[0]["X"].numpy()

    assert (pixels >= -1).all()
    assert (pixels <= 1).all()
    assert pixels.max() > 0.9


def test_single_target_audio_dataset_no_noise(
    single_target_audio_dataset_long_audio_df,
):
    dataset = SingleTargetAudioDataset(
        single_target_audio_dataset_long_audio_df, label_dict=None
    )
    rgb_image = dataset[0]["X"]
    channel_0 = rgb_image[0]
    channel_1 = rgb_image[1]
    channel_2 = rgb_image[2]
    assert_array_almost_equal(channel_0, channel_1)
    assert_array_almost_equal(channel_0, channel_2)
    assert_array_almost_equal(channel_1, channel_2)


def test_single_target_audio_dataset_with_noise(
    single_target_audio_dataset_long_audio_df,
):
    dataset = SingleTargetAudioDataset(
        single_target_audio_dataset_long_audio_df,
        label_dict={0: "hello"},
        add_noise=True,
    )
    rgb_image = dataset[0]["X"]
    channel_0 = rgb_image[0]
    channel_1 = rgb_image[1]
    channel_2 = rgb_image[2]
    assert_array_equal(channel_0, channel_1)
    assert_array_equal(channel_0, channel_2)
    assert_array_equal(channel_1, channel_2)
