#!/usr/bin/env python3
from opensoundscape.audio import Audio
import pytest
from pathlib import Path
from shutil import rmtree
from opensoundscape.datasets import Splitter, BinaryFromAudio
from torch.utils.data import DataLoader
import pandas as pd


tmp_path = "tests/_tmp_split"


@pytest.fixture()
def temporary_split_storage(request):
    path = Path(tmp_path)
    path.mkdir()

    def fin():
        path.rmdir()

    request.addfinalizer(fin)
    return path


@pytest.fixture()
def splitter_results_default(request):
    split0 = Path(f"{tmp_path}/5c5f4b5484945db37725533cd6a530f7.wav")
    split1 = Path(f"{tmp_path}/d174a640f6b3ed0cb42ca686576f663b.wav")

    def fin():
        split0.unlink()
        split1.unlink()

    request.addfinalizer(fin)
    return split0, split1


@pytest.fixture()
def splitter_results_last(request):
    split0 = Path(f"{tmp_path}/27f99f7d921cc464de465411f075e933.wav")
    split1 = Path(f"{tmp_path}/4202e0f72c56cb0763db165ffbbc8bc6.wav")

    def fin():
        split0.unlink()
        split1.unlink()

    request.addfinalizer(fin)
    return split0, split1


@pytest.fixture()
def one_min_audio_list():
    return [Path("tests/1min.wav")]


@pytest.fixture()
def binary_from_audio_df():
    return pd.read_csv("tests/input.csv")


def test_basic_splitting_operation_default(
    temporary_split_storage, splitter_results_default, one_min_audio_list
):
    dataset = Splitter(
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
        collate_fn=Splitter.collate_fn,
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
    dataset = Splitter(
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
        collate_fn=Splitter.collate_fn,
    )

    results = []
    for data in dataloader:
        for output in data:
            results.append(output)
    assert len(results) == 2

    split0, split1 = splitter_results_last
    assert split0.exists()
    assert split1.exists()


def test_binary_from_audio_default(binary_from_audio_df):
    dataset = BinaryFromAudio(binary_from_audio_df, height=225, width=226)
    assert dataset[0]["X"].shape == (3, 225, 226)
    assert dataset[0]["y"].shape == (1,)


def test_binary_from_audio_spec_augment(binary_from_audio_df):
    dataset = BinaryFromAudio(
        binary_from_audio_df, height=225, width=226, spec_augment=True
    )
    assert dataset[0]["X"].shape == (1, 225, 226)
    assert dataset[0]["y"].shape == (1,)
