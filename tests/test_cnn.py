from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.torch.datasets import AudioFileDataset
from opensoundscape.torch.loss import ResampleLoss
from opensoundscape.torch.models import cnn

from opensoundscape.torch.architectures.cnn_architectures import alexnet, resnet18
from opensoundscape.torch.architectures import cnn_architectures

from opensoundscape.helpers import make_clip_df
import pandas as pd
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shutil

import warnings


@pytest.fixture()
def model_save_path(request):
    path = Path("tests/models/temp.model")
    path.parent.mkdir(exist_ok=True)

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


@pytest.fixture()
def missing_file_df():
    return pd.DataFrame(index=["tests/audio/not_a_file.wav"])


def test_init_with_str():
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)


def test_train_single_target(train_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    model.single_target = True
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
    scores = model.predict(test_df)
    assert len(scores) == 2


def test_predict_on_list_of_files(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(test_df.index.values)
    assert len(scores) == 2


def test_predict_all_arch_4ch(test_df):
    for arch_name in cnn_architectures.ARCH_DICT.keys():
        try:
            arch = cnn_architectures.ARCH_DICT[arch_name](
                num_classes=2, num_channels=4, use_pretrained=False
            )
            if "inception" in arch_name:
                model = cnn.InceptionV3(
                    classes=[0, 1], sample_duration=5.0, sample_shape=[224, 224, 4]
                )
            else:
                model = cnn.CNN(
                    arch,
                    classes=[0, 1],
                    sample_duration=5.0,
                    sample_shape=[224, 224, 4],
                )
            scores = model.predict(test_df.index.values)
            assert len(scores) == 2
        except:
            raise Exception(f"{arch_name} failed")


def test_predict_all_arch_1ch(test_df):
    for arch_name in cnn_architectures.ARCH_DICT.keys():
        try:
            arch = cnn_architectures.ARCH_DICT[arch_name](
                num_classes=2, num_channels=1, use_pretrained=False
            )
            if "inception" in arch_name:
                model = cnn.InceptionV3(
                    classes=[0, 1], sample_duration=5.0, sample_shape=[224, 224, 4]
                )
            else:
                model = cnn.CNN(
                    arch,
                    classes=[0, 1],
                    sample_duration=5.0,
                    sample_shape=[224, 224, 1],
                )
            scores = model.predict(test_df.index.values)
            assert len(scores) == 2
        except:
            raise Exception(f"{arch_name} failed")


def test_predict_on_clip_df(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=1.0)
    clip_df, _ = make_clip_df(test_df.index.values[0:1], clip_duration=1.0)
    clip_df = clip_df.reset_index().set_index(["file", "start_time", "end_time"])
    scores = model.predict(clip_df)
    assert len(scores) == 10


def test_prediction_overlap(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    model.single_target = True
    scores = model.predict(test_df, overlap_fraction=0.5)

    assert len(scores) == 3


def test_multi_target_prediction(train_df, test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(test_df)

    assert len(scores) == 2


def test_predict_missing_file_is_unsafe_sample(missing_file_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(missing_file_df)

    assert len(scores) == 0

    # TODO get the unsafe samples from log file, then delete it
    # assert len(unsafe_samples) == 1


def test_predict_wrong_input_error(test_df):
    """cannot pass a preprocessor or dataset to predict. only file paths as list or df"""
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    pre = SpectrogramPreprocessor(2.0)
    with pytest.raises(AssertionError):
        model.predict(pre)
    with pytest.raises(AssertionError):
        ds = AudioFileDataset(test_df, pre)
        model.predict(ds)


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
    """test passing architecture object to CNN class"""
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


def test_train_on_clip_df(train_df):
    """
    test training a model when Audio files are long/unsplit
    and a dataframe provides clip-level labels. Training
    should internally load a relevant clip from the audio
    file and get its labels from the dataframe
    """
    model = cnn.CNN("resnet18", [0, 1], sample_duration=2)
    train_df, _ = make_clip_df(train_df.index.values, clip_duration=2)
    train_df = train_df.reset_index().set_index(["file", "start_time", "end_time"])
    train_df[0] = np.random.choice([0, 1], size=10)
    train_df[1] = np.random.choice([0, 1], size=10)
    model.train(
        train_df,
        train_df,
        save_path="tests/models/",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )


def test_predict_without_splitting(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(test_df, split_files_into_clips=False)
    assert len(scores) == len(test_df)


def test_predict_splitting_short_file(short_file_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scores = model.predict(short_file_df)
        assert len(scores) == 0
        assert "prediction_dataset" in str(w[0].message)


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


def test_save_load_and_train_model_resample_loss(train_df):
    arch = alexnet(2, use_pretrained=False)
    classes = [0, 1]

    m = cnn.CNN(arch, classes, 1.0)
    cnn.use_resample_loss(m)
    m.save("tests/models/saved1.model")
    m2 = cnn.load_model("tests/models/saved1.model")
    assert m2.classes == classes
    assert type(m2) == cnn.CNN

    assert m2.loss_cls == ResampleLoss

    # make sure it still trains ok after reloading w/resample loss
    m2.train(
        train_df,
        train_df,
        save_path="tests/models/",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )

    shutil.rmtree("tests/models/")


def test_prediction_warns_different_classes(train_df):
    model = cnn.CNN("resnet18", classes=["a", "b"], sample_duration=5.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # raises warning bc test_df columns != model.classes
        model.predict(train_df)
        all_warnings = ""
        for wi in w:
            all_warnings += str(wi.message)
        assert "classes" in all_warnings


def test_prediction_returns_consistent_values(train_df):
    model = cnn.CNN("resnet18", classes=["a", "b"], sample_duration=5.0)
    a = model.predict(train_df)
    b = model.predict(train_df)
    assert np.allclose(a.values, b.values, 1e-6)


def test_save_and_load_weights(model_save_path):
    arch = resnet18(2, use_pretrained=False)
    model = cnn.CNN("resnet18", classes=["a", "b"], sample_duration=5.0)
    model.save_weights(model_save_path)
    model1 = cnn.CNN(arch, classes=["a", "b"], sample_duration=5.0)
    model1.load_weights(model_save_path)
    assert np.array_equal(
        model.network.state_dict()["conv1.weight"].numpy(),
        model1.network.state_dict()["conv1.weight"].numpy(),
    )


def test_eval(train_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=2)
    scores = model.predict(train_df, split_files_into_clips=False)
    model.eval(train_df.values, scores.values)


def test_split_resnet_feat_clf(train_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=2)
    cnn.separate_resnet_feat_clf(model)
    assert "feature" in model.optimizer_params
    model.optimizer_params["feature"]["lr"] = 0.1
    model.train(train_df, epochs=0, save_path="tests/models")
    shutil.rmtree("tests/models/")


# test load_outdated_model?


def test_train_no_validation(train_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=2)
    model.train(train_df, save_path="tests/models")
    shutil.rmtree("tests/models/")
