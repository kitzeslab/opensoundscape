#!/usr/bin/env python3
from opensoundscape.audio import Audio
import pytest
from pathlib import Path
from shutil import rmtree
from opensoundscape.datasets import Splitter
from torch.utils.data import DataLoader


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
def splitter_results(request):
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


def test_basic_splitting_operation(
    temporary_split_storage, splitter_results, one_min_audio_list
):
    dataset = Splitter(
        one_min_audio_list,
        duration=30,
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

    split0, split1 = splitter_results
    assert split0.exists()
    assert split1.exists()
