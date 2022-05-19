from opensoundscape.preprocess.preprocessors import SpecPreprocessor
from opensoundscape.torch.models import cnn

from opensoundscape.torch.architectures.cnn_architectures import alexnet
import pandas as pd
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shutil


@pytest.fixture()
def model_save_path(request):
    path = Path("tests/models/temp.model")

    # always delete this at the end
    def fin():
        path.unlink()

    request.addfinalizer(fin)

    return path


@pytest.fixture()
def train_df():
    return pd.DataFrame(
        index=["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"],
        data=[[0, 1], [1, 0]],
    )


@pytest.fixture()
def test_df():
    return pd.DataFrame(index=["tests/audio/silence_10s.mp3"])


@pytest.fixture()
def short_file_df():
    return pd.DataFrame(index=["tests/audio/veryshort.wav"])


def test_init_with_str():
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)


def test_train_single_target(train_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    model.single_target = True
    model.train(  # TODO: is there a default overlay?
        train_df,
        train_df,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    shutil.rmtree("tests/models/")


def test_train_multi_target(train_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    shutil.rmtree("tests/models/")


def test_train_resample_loss(train_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    cnn.use_resample_loss(model)
    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    shutil.rmtree("tests/models/")


def test_train_one_class(train_df):
    model = cnn.CNN("resnet18", classes=[0], sample_duration=5.0)
    model.single_target = True
    model.train(
        train_df[[0]],
        train_df[[0]],
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    shutil.rmtree("tests/models/")


def test_single_target_prediction(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    model.single_target = True
    scores, preds, _ = model.predict(test_df, binary_preds="single_target")

    assert len(scores) == 2
    assert len(preds) == 2


def test_prediction_overlap(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    model.single_target = True
    scores, preds, _ = model.predict(
        test_df, binary_preds="single_target", overlap_fraction=0.5
    )

    assert len(scores) == 3
    assert len(preds) == 3


def test_multi_target_prediction(train_df, test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    _, preds, _ = model.predict(test_df, binary_preds="multi_target", threshold=0.1)
    scores, preds, _ = model.predict(
        test_df, binary_preds="multi_target", threshold=0.1
    )

    assert len(scores) == 2
    assert len(preds) == 2


def test_train_predict_inception(train_df):
    model = cnn.InceptionV3([0, 1], 5.0, use_pretrained=False)
    model.train(
        train_df,
        train_df,
        save_path="tests/models/",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_df, num_workers=0)
    shutil.rmtree("tests/models/")


def test_train_predict_architecture(train_df):
    """test passing a specific architecture to PytorchModel"""
    arch = alexnet(2, use_pretrained=False)
    model = cnn.CNN(arch, [0, 1], sample_duration=2)
    model.train(
        train_df,
        train_df,
        save_path="tests/models/",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_df, num_workers=0)
    shutil.rmtree("tests/models/")


def test_predict_without_splitting(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    scores, preds, _ = model.predict(
        test_df, split_files_into_clips=False, binary_preds="multi_target"
    )
    assert len(scores) == len(test_df)
    assert len(preds) == len(test_df)


def test_predict_splitting_short_file(short_file_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    scores, _, _ = model.predict(short_file_df)
    assert len(scores) == 0


def test_save_and_load_model(model_save_path):
    arch = alexnet(2, use_pretrained=False)
    classes = [0, 1]

    cnn.CNN(arch, classes, 1.0).save(model_save_path)
    m = cnn.load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == cnn.CNN

    cnn.InceptionV3(classes, 1.0, use_pretrained=False).save(model_save_path)
    m = cnn.load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == cnn.InceptionV3


# def test_save_and_load_weights():

# def test_eval

# def test make_samples

# def test_split_resnet_feat_clf
# test load_outdated_model?

# TODO: allow training w no validation? if so, its broken. if not, raise error.
