from opensoundscape.preprocess.preprocessors import (
    CnnPreprocessor,
    LongAudioPreprocessor,
    ClipLoadingSpectrogramPreprocessor,
)
from opensoundscape.torch.models.cnn import (
    PytorchModel,
    Resnet18Multiclass,
    Resnet18Binary,
    InceptionV3,
    load_model,
)
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
def train_dataset():
    df = pd.DataFrame(
        index=["tests/audio/great_plains_toad.wav", "tests/audio/1min.wav"],
        data=[[0, 1], [1, 0]],
        columns=["negative", "positive"],
    )
    return CnnPreprocessor(df, overlay_df=None)


@pytest.fixture()
def long_audio_dataset():
    df = pd.DataFrame(index=["tests/audio/1min.wav"])
    return LongAudioPreprocessor(
        df, audio_length=5.0, clip_overlap=0.0, out_shape=[224, 224]
    )


@pytest.fixture()
def clip_loading_preprocessor():
    import librosa
    from opensoundscape.helpers import generate_clip_times_df

    # prepare a df for clip loading preprocessor: start_time, end_time columns
    files = ["tests/audio/1min.wav"]
    clip_dfs = []
    for f in files:
        t = librosa.get_duration(filename=f)
        clips = generate_clip_times_df(t, 5, 0)
        clips.index = [f] * len(clips)
        clips.index.name = "file"
        clip_dfs.append(clips)
    clip_df = pd.concat(clip_dfs)
    return ClipLoadingSpectrogramPreprocessor(clip_df)


@pytest.fixture()
def test_dataset():
    df = pd.DataFrame(
        index=["tests/audio/great_plains_toad.wav", "tests/audio/1min.wav"]
    )
    return CnnPreprocessor(df, overlay_df=None, return_labels=False)


def test_multiclass_object_init():
    _ = Resnet18Multiclass([0, 1, 2, 3])


def test_init_with_str():
    model = PytorchModel("resnet18", classes=[0, 1])


def test_train(train_dataset):
    binary = Resnet18Binary(classes=["negative", "positive"])
    binary.train(
        train_dataset,
        train_dataset,
        save_path="tests/models/binary",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )


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


def test_single_target_prediction(train_dataset):
    binary = Resnet18Binary(classes=["negative", "positive"])
    _, preds, _ = binary.predict(train_dataset, binary_preds="single_target")
    assert np.sum(preds.iloc[0].values) == 1


def test_multi_target_prediction(train_dataset, test_dataset):
    binary = Resnet18Binary(classes=["negative", "positive"])
    _, preds, _ = binary.predict(
        test_dataset, binary_preds="multi_target", threshold=0.1
    )
    _, preds, _ = binary.predict(
        train_dataset, binary_preds="multi_target", threshold=0.1
    )
    assert int(np.sum(preds.iloc[0].values)) == 2


def test_train_predict_inception(train_dataset):
    model = InceptionV3(["negative", "positive"], use_pretrained=False)
    train_dataset_inception = train_dataset.sample(frac=1)
    # Inception expects input shape=(299,299)
    train_dataset_inception.actions.to_img.set(shape=[299, 299])
    model.train(
        train_dataset_inception,
        train_dataset_inception,
        save_path="tests/models/multiclass",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_dataset, num_workers=0)


def test_train_predict_architecture(train_dataset):
    """test passing a specific architecture to PytorchModel"""
    arch = alexnet(2, use_pretrained=False)
    model = PytorchModel(arch, ["negative", "positive"])
    model.train(
        train_dataset,
        train_dataset,
        save_path="tests/models/multiclass",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_dataset, num_workers=0)
    model_path = Path("tests/models/multiclass/best.model")
    model.save(model_path)
    assert model_path.exists()
    shutil.rmtree("tests/models/multiclass/")


def test_split_and_predict(long_audio_dataset):
    binary = Resnet18Binary(classes=["negative", "positive"])
    scores, preds, _ = binary.split_and_predict(
        long_audio_dataset, binary_preds="single_target"
    )
    assert len(scores) == 12
    assert len(preds) == 12


def test_predict_with_cliploading(clip_loading_preprocessor):
    binary = Resnet18Binary(classes=["negative", "positive"])
    scores, _, _ = binary.predict(clip_loading_preprocessor, binary_preds=None)
    assert len(scores) == 12


def test_save_and_load_model(model_save_path):
    arch = alexnet(2, use_pretrained=False)
    classes = ["negative", "positive"]

    PytorchModel(arch, classes=classes).save(model_save_path)
    m = load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == PytorchModel

    Resnet18Binary(classes=classes, use_pretrained=False).save(model_save_path)
    m = load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == Resnet18Binary

    Resnet18Multiclass(classes=classes, use_pretrained=False).save(model_save_path)
    m = load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == Resnet18Multiclass

    InceptionV3(classes=classes, use_pretrained=False).save(model_save_path)
    m = load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == InceptionV3
