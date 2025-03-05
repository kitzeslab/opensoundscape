import pandas as pd
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shutil
import torch
import torchmetrics

import warnings

import torchmetrics.classification

import opensoundscape
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.datasets import AudioFileDataset
from opensoundscape.ml.loss import ResampleLoss
from opensoundscape.ml import cnn
from opensoundscape.preprocess.utils import PreprocessingError

from opensoundscape.ml.cnn_architectures import alexnet, resnet18
from opensoundscape.ml import cnn_architectures

from opensoundscape.sample import AudioSample
from opensoundscape.ml.cam import CAM

from opensoundscape.utils import make_clip_df


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
def train_df_clips(train_df):
    clip_df = make_clip_df(train_df.index.values, clip_duration=1.0)
    clip_df[0] = np.random.choice([0, 1], size=len(clip_df))
    clip_df[1] = np.random.choice([0, 1], size=len(clip_df))
    return clip_df


@pytest.fixture()
def train_df_relative():
    clip_df = make_clip_df(
        ["silence_10s.mp3", "silence_10s.mp3"],
        clip_duration=1.0,
        audio_root="tests/audio/",
    )
    clip_df[0] = np.random.choice([0, 1], size=len(clip_df))
    clip_df[1] = np.random.choice([0, 1], size=len(clip_df))
    return clip_df


@pytest.fixture()
def test_df():
    return pd.DataFrame(index=["tests/audio/silence_10s.mp3"])


@pytest.fixture()
def short_file_df():
    return pd.DataFrame(index=["tests/audio/veryshort.wav"])


@pytest.fixture()
def missing_file_df():
    return pd.DataFrame(index=["tests/audio/not_a_file.wav"])


@pytest.fixture()
def onemin_wav_df():
    return pd.DataFrame(index=["tests/audio/1min.wav"])


def test_init_with_str():
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)


def test_save_load():
    classes = [0, 1]
    arch = resnet18(2, weights=None, num_channels=1)
    m = cnn.SpectrogramClassifier(architecture=arch, classes=classes, sample_duration=3)
    m.save("tests/models/saved1.model")
    m2 = cnn.SpectrogramClassifier.load("tests/models/saved1.model")
    assert m2.classes == classes
    assert type(m2) == cnn.SpectrogramClassifier
    assert m2.preprocessor.sample_duration == 3

    # use class name and class look-up dictionary to re-create the correct class from saved model
    # and "class" key
    m3 = cnn.load_model("tests/models/saved1.model")
    assert m3.classes == classes
    assert type(m3) == cnn.SpectrogramClassifier
    assert m3.preprocessor.sample_duration == 3

    # check that the weights are equivalent
    for k in m.network.state_dict().keys():
        assert np.allclose(
            m.network.state_dict()[k].numpy(), m2.network.state_dict()[k].numpy()
        )


def test_save_load_pickel(train_df):
    """when saving with pickle, can resume training and have the same optimizer state"""
    classes = [0, 1]
    m = cnn.SpectrogramClassifier(
        architecture="resnet18", classes=classes, sample_duration=3
    )
    m.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    shutil.rmtree("tests/models/")
    m.save("tests/models/saved1.model", pickle=True)
    m2 = cnn.SpectrogramClassifier.load("tests/models/saved1.model")
    assert m2.classes == classes
    assert type(m2) == cnn.SpectrogramClassifier
    assert str(m.scheduler.state_dict()) == str(m2.scheduler.state_dict())
    assert str(m.optimizer.state_dict()) == str(m2.optimizer.state_dict())
    assert m2.preprocessor.sample_duration == 3


def test_train_single_target(train_df):
    model = cnn.CNN(
        architecture="resnet18", classes=[0, 1], sample_duration=5.0, single_target=True
    )
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
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
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


def test_train_on_clip_df(train_df_clips):
    """
    test training a model when Audio files are long/unsplit
    and a dataframe provides clip-level labels. Training
    should internally load a relevant clip from the audio
    file and get its labels from the dataframe
    """
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=1.0)
    model.train(
        train_df_clips,
        train_df_clips,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    shutil.rmtree("tests/models/")


def test_train_with_audio_root(train_df_relative):
    """
    test training a model when Audio files are long/unsplit
    and a dataframe provides clip-level labels. Training
    should internally load a relevant clip from the audio
    file and get its labels from the dataframe
    """
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=1.0)
    model.train(
        train_df_relative,
        train_df_relative,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
        audio_root="tests/audio",
    )
    shutil.rmtree("tests/models/")


def test_classifier_custom_lr(train_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    model.optimizer_params["kwargs"]["lr"] = 0.001
    model.optimizer_params["classifier_lr"] = 0.02
    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=0,
    )
    assert model.optimizer.param_groups[0]["lr"] == 0.001
    assert next(model.network.parameters()) in model.optimizer.param_groups[0]["params"]
    assert model.optimizer.param_groups[1]["lr"] == 0.02
    assert (
        next(model.classifier.parameters()) in model.optimizer.param_groups[1]["params"]
    )


def test_reset_or_keep_optimizer_and_scheduler(train_df):
    import copy
    from opensoundscape.utils import set_seed

    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)

    set_seed(0)
    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )

    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=0,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    opt1 = copy.deepcopy(model.optimizer)

    # test that default (reset_optimzer=False, restart_scheduler=False)
    # retains optimizer and scheduler state from previous .train()
    # rather than re-initializing with random values
    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=0,
        batch_size=2,
        save_interval=10,
        num_workers=0,
        reset_optimizer=False,
        restart_scheduler=False,
    )

    assert (
        model.optimizer.state_dict()["state"][0]["momentum_buffer"]
        == opt1.state_dict()["state"][0]["momentum_buffer"]
    ).all()
    assert model.scheduler.state_dict()["last_epoch"] == 1
    assert model.scheduler.state_dict()["_step_count"] == 2

    # test that reset_optimizer=True, restart_scheduler=False
    # resets the optimizer to the initial state
    set_seed(0)
    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=0,
        batch_size=2,
        save_interval=10,
        num_workers=0,
        reset_optimizer=True,
        restart_scheduler=True,
    )

    assert model.optimizer.state_dict()["state"] == {}
    assert model.scheduler.state_dict()["last_epoch"] == 0
    assert model.scheduler.state_dict()["_step_count"] == 1

    shutil.rmtree("tests/models/")


def test_train_amp_cpu(train_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    # first test with cpu
    model.device = "cpu"
    model.use_amp = True
    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_df)
    shutil.rmtree("tests/models/")


def test_train_amp_mps(train_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    # if cuda is available, test with cuda
    if torch.cuda.is_available():
        assert model.device.type == "cuda"
    else:
        return  # cannot test cuda
    model.use_amp = True
    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_df)
    shutil.rmtree("tests/models/")


def test_train_amp_mps(train_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    if torch.mps.is_available():
        assert model.device.type == "mps"
    else:
        return  # cannot test mps on this machine
    model.use_amp = True
    model.train(
        train_df,
        train_df,
        save_path="tests/models",
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_df)
    shutil.rmtree("tests/models/")


def test_train_resample_loss(train_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    cnn.use_resample_loss(model, train_df=train_df)
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
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0],
        sample_duration=5.0,
    )
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


def test_single_target_setter():
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    assert model.single_target is False
    assert model._single_target == False
    assert isinstance(model.loss_fn, torch.nn.BCEWithLogitsLoss)
    assert isinstance(
        model.torch_metrics["map"],
        torchmetrics.classification.MultilabelAveragePrecision,
    )
    # use setter of property
    model.single_target = True
    assert model.single_target is True
    assert model._single_target == True
    assert isinstance(model.loss_fn, torch.nn.CrossEntropyLoss)
    assert isinstance(
        model.torch_metrics["map"],
        torchmetrics.classification.MulticlassAveragePrecision,
    )
    # switch back to multi-target
    model.single_target = False
    assert model.single_target is False
    assert model._single_target == False
    assert isinstance(model.loss_fn, torch.nn.BCEWithLogitsLoss)
    assert isinstance(
        model.torch_metrics["map"],
        torchmetrics.classification.MultilabelAveragePrecision,
    )


def test_single_target_prediction(train_df_clips):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=1.0)
    model.single_target = True
    scores = model.predict(train_df_clips)
    assert len(scores) == 20
    model.eval(train_df_clips.values, scores.values)


def test_predict_on_list_of_files(test_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(test_df.index.values)
    assert len(scores) == 2


def test_predict_with_audio_root():
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(["silence_10s.mp3"], audio_root="tests/audio/")
    assert len(scores) == 2


def test_predict_and_embed_on_df_with_file_index(train_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(train_df)
    emb = model.embed(train_df)
    assert len(scores) == 4
    assert len(emb) == 4


def test_predict_on_empty_list():
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict([])
    expected = ["file", "start_time", "end_time", 0, 1]
    assert list(scores.reset_index().columns) == expected

    scores = model.predict([], split_files_into_clips=False)
    expected = ["index", 0, 1]
    assert list(scores.reset_index().columns) == expected


def test_predict_all_arch_4ch(test_df):
    for arch_name in cnn_architectures.ARCH_DICT.keys():
        try:
            arch = cnn_architectures.ARCH_DICT[arch_name](
                num_classes=2, num_channels=4, weights=None
            )
            if "inception" in arch_name:
                # inception requires 3 channels
                # (_transform_input() implementation is hard-coded for 3 channels)
                continue
            else:
                model = cnn.CNN(
                    architecture=arch,
                    classes=[0, 1],
                    sample_duration=5.0,
                    channels=4,
                )
            scores = model.predict(test_df.index.values)
            assert len(scores) == 2
        except Exception as e:
            raise Exception(f"{arch_name} failed") from e


def test_predict_all_arch_1ch(test_df):
    for arch_name in cnn_architectures.ARCH_DICT.keys():
        try:
            arch = cnn_architectures.ARCH_DICT[arch_name](
                num_classes=2, num_channels=1, weights=None
            )
            if "inception" in arch_name:
                # inception requires 3 channels
                continue
            else:
                model = cnn.CNN(
                    architecture=arch, classes=[0, 1], sample_duration=5.0, channels=1
                )
            scores = model.predict(test_df.index.values)
            assert len(scores) == 2
        except Exception as e:
            raise Exception(f"{arch_name} failed") from e


def test_predict_on_clip_df(test_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=1.0)
    clip_df = make_clip_df(test_df.index.values[0:1], clip_duration=1.0)
    scores = model.predict(clip_df)
    assert len(scores) == 10


def test_prediction_overlap(test_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    model.single_target = True
    scores = model.predict(test_df, overlap_fraction=0.5)

    assert len(scores) == 3


def test_predict_on_one_file(test_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=10)
    p = test_df.index.values[0]
    scores = model.predict(p)
    assert len(scores) == 1
    scores = model.predict(Path(p))
    assert len(scores) == 1


def test_multi_target_prediction(train_df, test_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(test_df)

    assert len(scores) == 2


def test_predict_missing_file_is_invalid_sample(missing_file_df, test_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)

    with pytest.raises(IndexError):
        # if all samples are invalid, will give IndexError
        model.predict(missing_file_df)

    scores, invalid_samples = model.predict(
        pd.concat([missing_file_df, test_df.head(1)]), return_invalid_samples=True
    )
    assert (
        len(scores) == 3
    )  # should have one row with nan values for the invalid sample
    isnan = lambda x: x != x
    assert np.all([isnan(score) for score in scores.iloc[0].values])
    assert len(invalid_samples) == 1
    assert missing_file_df.index.values[0] in invalid_samples


def test_predict_wrong_input_error(test_df):
    """cannot pass a preprocessor or dataset to predict. only file paths as list or df"""
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    pre = SpectrogramPreprocessor(2.0)
    with pytest.raises(AssertionError):
        model.predict(pre)
    with pytest.raises(AssertionError):
        ds = AudioFileDataset(test_df, pre)
        model.predict(ds)


def test_train_predict_inception(train_df):
    model = cnn.InceptionV3([0, 1], 5.0, weights=None)
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
    """test passing architecture object to CNN class

    should internally update `channels` to match architecture (3 channels)
    """
    for num_channels in (1, 3):
        arch = alexnet(2, weights=None, num_channels=num_channels)
        model = cnn.CNN(architecture=arch, classes=[0, 1], sample_duration=2)
        model.predict(train_df, num_workers=0)
        assert model.preprocessor.channels == num_channels


def test_train_bad_index(train_df):
    """
    AssertionError catches case where index is not one of the allowed formats
    """
    model = cnn.CNN("resnet18", [0, 1], sample_duration=2)
    # reset the index so that train_df index is integers (not an allowed format)
    train_df = make_clip_df(train_df.index.values, clip_duration=2).reset_index()
    train_df[0] = np.random.choice([0, 1], size=10)
    train_df[1] = np.random.choice([0, 1], size=10)
    with pytest.raises(AssertionError):
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
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(test_df, split_files_into_clips=False)
    assert len(scores) == len(test_df)


def test_predict_splitting_short_file(short_file_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scores = model.predict(short_file_df)
        assert len(scores) == 0
        all_warnings = ""
        for wi in w:
            all_warnings += str(wi.message)
        assert "prediction_dataset" in all_warnings


def test_save_and_load_model(model_save_path):
    classes = [0, 1]

    cnn.CNN(architecture="alexnet", classes=classes, sample_duration=1.0).save(
        model_save_path
    )
    m = cnn.load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == cnn.CNN

    cnn.InceptionV3(classes=classes, sample_duration=1.0, weights=None).save(
        model_save_path
    )
    m = cnn.load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == cnn.InceptionV3


def test_save_and_load_model_custom_arch(model_save_path):
    # using a custom architecture: define a generator function that takes
    # num_classes and num_channels as arguments and returns the architecture
    # then register it with the cnn_architectures ARCH_DICT
    classes = [0, 1]

    @cnn_architectures.register_arch
    def my_alexnet_generator(
        num_classes, num_channels, weights="DEFAULT", freeze_feature_extractor=False
    ):
        # for example, we ignore the freeze_feature_extractor argument in the generator
        return alexnet(num_classes, num_channels=num_channels, weights=weights)

    arch = my_alexnet_generator(2, 1)
    arch.constructor_name = "my_alexnet_generator"
    assert arch.constructor_name in cnn_architectures.ARCH_DICT
    m = cnn.CNN(architecture=arch, classes=classes, sample_duration=1.0)
    m.save(model_save_path)
    m2 = cnn.load_model(model_save_path)
    assert type(m2.network) == type(arch)

    # remove the custom architecture from the ARCH_DICT when done
    del cnn_architectures.ARCH_DICT["my_alexnet_generator"]


def test_init_positional_args():
    cnn.CNN("resnet18", [0, 1], 0)


def test_save_load_and_train_model_resample_loss(train_df):
    arch = alexnet(2, weights=None)
    classes = [0, 1]

    m = cnn.CNN(architecture=arch, classes=classes, sample_duration=1.0)
    cnn.use_resample_loss(m, train_df)
    m.save("tests/models/saved1.model", pickle=True)
    m2 = cnn.load_model("tests/models/saved1.model")
    assert m2.classes == classes
    assert type(m2) == cnn.CNN
    assert isinstance(m2.loss_fn, ResampleLoss)

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
    model = cnn.CNN(architecture="resnet18", classes=["a", "b"], sample_duration=5.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # raises warning bc test_df columns != model.classes
        model.predict(train_df)
        all_warnings = ""
        for wi in w:
            all_warnings += str(wi.message)
        assert "classes" in all_warnings


def test_train_raises_wrong_class_list(train_df):
    model = cnn.CNN(architecture="resnet18", classes=["different"], sample_duration=5.0)
    with pytest.raises(AssertionError):
        # raises AssertionError bc test_df columns != model.classes
        model.train(train_df)


def test_train_raises_labels_outside_range(train_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    train_df.iat[0, 0] = 2
    with pytest.raises(AssertionError):
        # raises AssertionError bc values outside [0,1] not allowed
        model.train(train_df)


def test_prediction_returns_consistent_values(train_df):
    model = cnn.CNN(architecture="resnet18", classes=["a", "b"], sample_duration=5.0)
    a = model.predict(train_df)
    b = model.predict(train_df)
    assert np.allclose(a.values, b.values, 1e-6)


def test_save_and_load_weights(model_save_path):
    arch = resnet18(2, weights=None, num_channels=1)
    model = cnn.CNN(architecture="resnet18", classes=["a", "b"], sample_duration=5.0)
    model.save_weights(model_save_path)
    model1 = cnn.CNN(arch, classes=["a", "b"], sample_duration=5.0)
    model1.load_weights(model_save_path)
    assert np.array_equal(
        model.network.state_dict()["conv1.weight"].numpy(),
        model1.network.state_dict()["conv1.weight"].numpy(),
    )


def test_eval(train_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=2)
    scores = model.predict(train_df, split_files_into_clips=False)
    model.eval(train_df.values, scores.values)


def test_eval_raises_bad_labels(train_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=2)
    scores = model.predict(train_df, split_files_into_clips=False)
    train_df.iat[0, 0] = 2
    with pytest.raises(AssertionError):
        # raises AssertionError bc values outside [0,1] not allowed
        model.eval(train_df.values, scores.values)


def test_train_no_validation(train_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=2)
    model.train(train_df, save_path="tests/models")
    shutil.rmtree("tests/models/")


def test_train_raise_errors(short_file_df, missing_file_df):
    files_df = pd.concat(
        [short_file_df, missing_file_df]
    )  # use 2 files. 1 file wrong is manually caught and userwarning raised
    files_df["class"] = [0, 1]  # add labels for training
    model = cnn.CNN("resnet18", classes=["class"], sample_duration=2)
    with pytest.raises(PreprocessingError):
        model.train(files_df, raise_errors=True)


def test_predict_raise_errors(short_file_df, onemin_wav_df):
    files_df = pd.concat(
        [short_file_df, onemin_wav_df]
    )  # use 2 files. 1 file wrong is manually caught and userwarning raised
    model = cnn.CNN(architecture="resnet18", classes=["class"], sample_duration=30)
    model.preprocessor.pipeline.bandpass.bypass = False  # ensure bandpass happens
    # add a bad param. this should be min_f
    model.preprocessor.pipeline.bandpass.params["low"] = 1

    with pytest.raises(PreprocessingError):
        model.predict(files_df, raise_errors=True)


def test_generate_cams(test_df):
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    samples = model.generate_cams(test_df)
    assert len(samples) == 2
    assert type(samples[0]) == AudioSample
    assert type(samples[0].cam) == CAM
    assert type(samples[0].cam.activation_maps) == pd.Series
    assert samples[0].cam.gbp_maps is None

    samples = model.generate_cams(test_df, guided_backprop=True, method=None)
    assert type(samples[0].cam.gbp_maps) == pd.Series
    assert samples[0].cam.activation_maps is None


def test_generate_samples(test_df):
    """should return list of AudioSample objects"""
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    samples = model.generate_samples(test_df)
    assert len(samples) == 2
    assert type(samples[0]) == AudioSample
    assert type(samples[0].data) == torch.Tensor
    assert type(samples[0].labels) == pd.Series


def test_generate_cams_batch_size(test_df):
    """smoke test for batch size > 1"""
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    _ = model.generate_cams(test_df, batch_size=2)


def test_generate_cams_num_workers(test_df):
    """smoke test for num workers > 1"""
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    _ = model.generate_cams(test_df, num_workers=2)


def test_generate_cams_scorecam_devices(test_df):
    """In pytorch_grad_cam <1.5.0 scorecam had device mismatch"""

    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    import pytorch_grad_cam

    _ = model.generate_cams(
        test_df,
        method=pytorch_grad_cam.ScoreCAM,
    )

    # very slow on cpu - but can uncomment to check
    # model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    # model.device = "cpu"
    # import pytorch_grad_cam

    # _ = model.generate_cams(
    #     test_df,
    #     method=pytorch_grad_cam.ScoreCAM,
    # )


def test_generate_cams_methods(test_df):
    """test each supported method both by passing class and string name"""

    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    import pytorch_grad_cam

    methods_dict = {
        "gradcam": pytorch_grad_cam.GradCAM,
        "hirescam": pytorch_grad_cam.HiResCAM,
        "scorecam": pytorch_grad_cam.ScoreCAM,
        "gradcam++": pytorch_grad_cam.GradCAMPlusPlus,
        "ablationcam": pytorch_grad_cam.AblationCAM,
        "xgradcam": pytorch_grad_cam.XGradCAM,
        "eigencam": pytorch_grad_cam.EigenCAM,
        "eigengradcam": pytorch_grad_cam.EigenGradCAM,
        "layercam": pytorch_grad_cam.LayerCAM,
        "fullgrad": pytorch_grad_cam.FullGrad,
        "gradcamelementwise": pytorch_grad_cam.GradCAMElementWise,
    }
    # use each class
    for method_cls in methods_dict.values():
        _ = model.generate_cams(test_df, method=method_cls)

    # use each method's string name
    for method_str in methods_dict.keys():
        _ = model.generate_cams(test_df, method=method_str)


def test_generate_cam_all_architectures(test_df):
    for arch_name in cnn_architectures.ARCH_DICT.keys():
        try:
            arch = cnn_architectures.ARCH_DICT[arch_name](
                num_classes=2, num_channels=1, weights=None
            )
            model = cnn.CNN(
                architecture=arch, classes=[0, 1], sample_duration=5.0, channels=1
            )
            _ = model.generate_cams(test_df.head(1))
        except Exception as e:
            raise Exception(f"{arch_name} failed") from e


def test_generate_cams_target_layers(test_df):
    """specify multiple target layers for cam"""
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    _ = model.generate_cams(
        test_df, target_layers=[model.network.layer3, model.network.layer4]
    )


def test_train_with_posixpath(train_df):
    """test that train works with pathlib.Path objects"""
    from pathlib import Path

    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)

    # turn the file paths into Path objects.
    posix_paths = [Path(p) for p in train_df.index]

    # change the index of train_df to be the Path objects
    train_df.index = posix_paths

    model.train(
        train_df,
        train_df,
        save_path=Path("tests/models"),
        epochs=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )

    shutil.rmtree("tests/models/")


def test_predict_posixpath_missing_files(missing_file_df, test_df):
    """Test that predict works with pathlib.Path objects"""
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)

    missing_file_df.index = [Path(p) for p in missing_file_df.index]
    test_df.index = [Path(p) for p in test_df.index]
    with pytest.raises(IndexError):
        # if all samples are invalid, will give IndexError
        model.predict(missing_file_df)

    scores, invalid_samples = model.predict(
        pd.concat([missing_file_df, test_df.head(1)]), return_invalid_samples=True
    )
    assert (
        len(scores) == 3
    )  # should have one row with nan values for the invalid sample
    isnan = lambda x: x != x
    assert np.all([isnan(score) for score in scores.iloc[0].values])
    assert len(invalid_samples) == 1
    assert missing_file_df.index.values[0] in invalid_samples


def test_predict_overlap_fraction_deprecated(test_df):
    """
    should give deprecation error if clip_overlap_fraction is passed.

    Future version will remove this argument in favor of clip_overlap_fraction

    also, should raise AssertionError if both args are passed (over-specified)
    """
    model = cnn.CNN(architecture="resnet18", classes=[0, 1], sample_duration=5.0)
    with pytest.warns(DeprecationWarning):
        scores = model.predict(test_df, overlap_fraction=0.5)
        assert len(scores) == 3
    with pytest.raises(AssertionError):
        model.predict(test_df, overlap_fraction=0.5, clip_overlap_fraction=0.5)


def test_embed(test_df):
    from opensoundscape.ml.cnn_architectures import list_architectures

    for arch in list_architectures():
        if "inception" in arch:
            continue
        try:
            m = cnn.SpectrogramClassifier(
                classes=[0],
                single_target=False,
                architecture=arch,
                sample_duration=5,
            )
            embeddings = m.embed(samples=test_df, avgpool=True, progress_bar=False)
            assert embeddings.shape[0] == 2
            assert len(embeddings.shape) == 2
            assert isinstance(embeddings, pd.DataFrame)
        except Exception as e:
            raise Exception(f"{arch} failed") from e


def test_embed_no_avgpool(test_df):
    # returns arrays rather than dataframes
    m = cnn.SpectrogramClassifier(
        classes=[0, 1],
        single_target=False,
        architecture="resnet18",
        sample_duration=5,
    )
    embeddings = m.embed(
        samples=test_df,
        avgpool=False,
        progress_bar=False,
        target_layer=m.network.layer4,
    )
    assert embeddings.shape == (2, 512, 7, 7)


def test_embed_return_array(test_df):
    # returns arrays rather than dataframes
    m = cnn.SpectrogramClassifier(
        classes=[0, 1],
        single_target=False,
        architecture="resnet18",
        sample_duration=5,
    )
    embeddings = m.embed(
        samples=test_df,
        progress_bar=False,
        target_layer=m.network.layer4,
        return_dfs=False,
    )
    assert embeddings.shape == (2, 512)
    assert isinstance(embeddings, np.ndarray)


def test_embed_one_sample(train_df):
    m = cnn.SpectrogramClassifier(
        classes=[0, 1, 2],
        single_target=False,
        architecture="resnet18",
        sample_duration=10,
    )
    embeddings = m.embed(samples=train_df.head(1), avgpool=True, progress_bar=False)
    assert embeddings.shape == (1, 512)


def test_call_with_intermediate_layers(test_df):
    """test that passing intermediate_layers to SpectrogramClassifier.__call__ returns tensors of expected shape"""
    model = cnn.SpectrogramClassifier(
        architecture="resnet18", classes=[0, 1], sample_duration=5.0
    )
    dl = model.predict_dataloader(test_df)
    preds, intermediate_outs = model(
        dl, intermediate_layers=[model.network.layer1, model.network.layer4]
    )
    assert len(intermediate_outs) == 2
    assert np.shape(intermediate_outs[0]) == (2, 64)
    assert np.shape(intermediate_outs[1]) == (2, 512)
    preds, intermediate_outs = model(
        dl, intermediate_layers=[model.network.layer4], avgpool_intermediates=False
    )
    assert len(intermediate_outs) == 1
    assert np.shape(intermediate_outs[0]) == (2, 512, 7, 7)


def test_freeze_layers_except_and_unfreeze():
    model = cnn.SpectrogramClassifier(
        architecture="resnet18", classes=[0, 1], sample_duration=5.0
    )
    model.freeze_layers_except()
    for param in model.network.parameters():
        assert not param.requires_grad

    model.unfreeze()
    for param in model.network.parameters():
        assert param.requires_grad

    model.freeze_layers_except(model.network.layer1)
    for name, param in model.network.named_parameters():
        if "layer1" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad

    # should also work when the weights were previously frozen
    model.freeze_layers_except()
    model.freeze_layers_except(model.network.layer1)
    for name, param in model.network.named_parameters():
        if "layer1" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad


def test_freeze_feature_extractor_all_arch():
    """freeze_feature_extractor() should result in only the classifier layer having requires_grad=True

    all other params will have requires_grad=False. Classifier layer is determined by .classifier property
    """
    for arch_name in cnn_architectures.ARCH_DICT.keys():
        try:
            arch = cnn_architectures.ARCH_DICT[arch_name](
                num_classes=2, num_channels=1, weights=None
            )
            model = cnn.CNN(
                architecture=arch, classes=[0, 1], sample_duration=5.0, channels=1
            )
            model.freeze_feature_extractor()

            clf_params = list([id(x) for x in model.classifier.parameters()])
            for p in model.network.parameters():
                if id(p) in clf_params:
                    assert p.requires_grad
                else:
                    assert not p.requires_grad
        except Exception as e:
            raise Exception(f"{arch_name} failed") from e


def test_change_classes_all_arch():
    """change_classes should change the classes attribute and the output layer of the network"""
    for arch_name in cnn_architectures.ARCH_DICT.keys():
        try:
            model = cnn.CNN(
                architecture=arch_name, classes=[0, 1], sample_duration=5.0, channels=1
            )
            if arch_name == "squeezenet1_0":
                # has conv2d not linear layer
                with pytest.raises(AssertionError):
                    model.change_classes([0, 1, 2])
            else:
                model.change_classes([0, 1, 2])
                assert model.classes == [0, 1, 2]
                assert model.classifier.out_features == 3
        except Exception as e:
            raise Exception(f"{arch_name} failed") from e
