from opensoundscape.preprocess.preprocessors import CnnPreprocessor
from opensoundscape.torch.models.cnn import (
    PytorchModel,
    Resnet18Multiclass,
    Resnet18Binary,
)

import pandas as pd
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shutil


@pytest.fixture()
def train_dataset():
    df = pd.DataFrame(
        index=["tests/audio/great_plains_toad.wav", "tests/audio/1min.wav"],
        data=[[0, 1], [1, 0]],
        columns=["negative", "positive"],
    )
    return CnnPreprocessor(df, overlay_df=None)


def test_multiclass_object_init():
    _ = Resnet18Multiclass([0, 1, 2, 3])


def test_train(train_dataset):
    binary = Resnet18Binary()
    binary.train(
        train_dataset,
        train_dataset,
        save_path="tests/models/binary",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model_path = Path("tests/models/binary/best.model")
    binary.save(model_path)
    assert model_path.exists()
    shutil.rmtree("tests/models/binary")


def test_train_multiclass(train_dataset):
    model = Resnet18Multiclass(["negative", "positive"])
    model.train(
        train_dataset,
        train_dataset,
        save_path="tests/models/multiclass",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model_path = Path("tests/models/multiclass/best.model")
    model.save(model_path)
    assert model_path.exists()
    shutil.rmtree("tests/models/multiclass/")


def test_single_target_prediction(train_dataset):
    binary = Resnet18Binary()
    _, preds, _ = binary.predict(train_dataset, binary_preds="single_target")
    assert np.sum(preds.iloc[0].values) == 1


def test_multi_target_prediction(train_dataset):
    binary = Resnet18Binary()
    _, preds, _ = binary.predict(
        train_dataset, binary_preds="multi_target", threshold=0.1
    )
    assert int(np.sum(preds.iloc[0].values)) == 2
