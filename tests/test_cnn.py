import importlib.util
import pandas as pd
from pathlib import Path
import types

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
from opensoundscape.ml.shallow_classifier import MLPClassifier

from opensoundscape.sample import AudioSample
from opensoundscape.ml.cam import CAM

from opensoundscape.utils import make_clip_df, identity


@pytest.fixture()
def model_save_path(request, tmp_path):
    """Fixture providing a temporary model save path with proper cleanup.

    Uses pytest's tmp_path fixture to create isolated temporary directories
    for each test, ensuring no conflicts between parallel test runs.
    """
    path = tmp_path / "temp.model"

    # Cleanup function to remove the file if it exists
    def fin():
        if path.exists():
            path.unlink()

    request.addfinalizer(fin)

    return path


@pytest.fixture()
def onnx_save_path(request, tmp_path):
    """Fixture providing a temporary ONNX model save path with proper cleanup.

    Uses pytest's tmp_path fixture to create isolated temporary directories
    for each test, ensuring no conflicts between parallel test runs.
    """
    path = tmp_path / "temp.onnx"

    # Cleanup function to remove the file if it exists
    def fin():
        if path.exists():
            path.unlink()

    request.addfinalizer(fin)

    return path


@pytest.fixture()
def temp_model_dir(request, tmp_path):
    """Fixture providing a temporary directory for model saving with proper cleanup.

    This fixture is used for tests that need to save models to a directory
    rather than a specific file path.
    """
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)

    # Cleanup function to remove the directory and its contents
    def fin():
        if model_dir.exists():
            import shutil

            shutil.rmtree(model_dir)

    request.addfinalizer(fin)

    return model_dir


@pytest.fixture()
def train_df():
    return pd.DataFrame(
        index=["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"],
        data=[[0, 1], [1, 0]],
    )


@pytest.fixture()
def silence10s_path():
    return "tests/audio/silence_10s.mp3"


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


def are_state_dicts_close(sd1, sd2):
    if sd1.keys() != sd2.keys():
        return False
    return all(torch.allclose(sd1[k], sd2[k]) for k in sd1)


def test_init_with_str():
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )


def test_save_load(model_save_path):
    classes = [0, 1]
    arch = resnet18(2, weights=None, num_channels=1)
    m = cnn.SpectrogramClassifier(
        architecture=arch,
        classes=classes,
        sample_duration=3,
        sample_rate=22050,
        arch_weights=None,
    )
    m.save(model_save_path)
    m2 = cnn.SpectrogramClassifier.load(model_save_path)
    assert m2.classes == classes
    assert type(m2) == cnn.SpectrogramClassifier
    assert m2.preprocessor.sample_duration == 3

    # use class name and class look-up dictionary to re-create the correct class from saved model
    # and "class" key
    m3 = cnn.load_model(model_save_path)
    assert m3.classes == classes
    assert type(m3) == cnn.SpectrogramClassifier
    assert m3.preprocessor.sample_duration == 3

    # check that the weights are equivalent
    assert are_state_dicts_close(m.network.state_dict(), m2.network.state_dict())


def test_save_load_pickel(train_df, model_save_path, temp_model_dir):
    """when saving with pickle, can resume training and have the same optimizer state"""
    classes = [0, 1]
    m = cnn.SpectrogramClassifier(
        architecture="resnet18",
        classes=classes,
        sample_duration=3,
        sample_rate=22050,
        arch_weights=None,
    )
    m.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    # No need to manually remove directory - fixture handles cleanup
    m.save(model_save_path, pickle=True)
    m2 = cnn.SpectrogramClassifier.load(model_save_path)
    assert m2.classes == classes
    assert type(m2) == cnn.SpectrogramClassifier
    assert str(m.scheduler.state_dict()) == str(m2.scheduler.state_dict())
    assert str(m.optimizer.state_dict()) == str(m2.optimizer.state_dict())
    assert m2.preprocessor.sample_duration == 3


def test_train_single_target(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        single_target=True,
        arch_weights=None,
    )
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    # No need to manually remove directory - fixture handles cleanup


def test_train_wandb(train_df, temp_model_dir):
    # Use disabled mode so this test never depends on network/credentials.
    try:
        import wandb

        session = wandb.init(mode="disabled", reinit=True)
    except Exception:
        pytest.skip("Could not init wandb session")

    model = cnn.CNN(
        architecture="resnet18", classes=[0, 1], sample_duration=5.0, sample_rate=None
    )
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
        wandb_session=session,
    )
    session.finish()

    # clean up wandb files
    import os

    if os.path.exists("wandb"):
        shutil.rmtree("wandb")


onnx_deps = pytest.mark.skipif(
    not all(
        importlib.util.find_spec(pkg) is not None
        for pkg in ("onnx", "onnxruntime", "onnxscript")
    ),
    reason="onnx, onnxruntime, or onnxscript not installed",
)


def test_train_multi_target(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    # No need to manually remove directory - fixture handles cleanup


def test_train_on_clip_df(train_df_clips, temp_model_dir):
    """
    test training a model when Audio files are long/unsplit
    and a dataframe provides clip-level labels. Training
    should internally load a relevant clip from the audio
    file and get its labels from the dataframe
    """
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=1.0,
        sample_rate=22050,
        arch_weights=None,
    )
    model.train(
        train_df_clips,
        train_df_clips,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    # No need to manually remove directory - fixture handles cleanup


def test_train_with_audio_root(train_df_relative, temp_model_dir):
    """
    test training a model when Audio files are long/unsplit
    and a dataframe provides clip-level labels. Training
    should internally load a relevant clip from the audio
    file and get its labels from the dataframe
    """
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=1.0,
        sample_rate=22050,
        arch_weights=None,
    )
    model.train(
        train_df_relative,
        train_df_relative,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
        audio_root="tests/audio",
    )
    # No need to manually remove directory - fixture handles cleanup


def test_classifier_custom_lr(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    model.optimizer_params["kwargs"]["lr"] = 0.001
    model.optimizer_params["classifier_lr"] = 0.02
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=0,
    )
    # note: when using CosineAnnealingWithWarmupScheduler, 'lr' is the starting lr (very small), while 'initial_lr' is actually the peak lr after warmup
    assert model.optimizer.param_groups[0]["initial_lr"] == 0.001
    assert next(model.network.parameters()) in model.optimizer.param_groups[0]["params"]
    assert model.optimizer.param_groups[1]["initial_lr"] == 0.02
    assert (
        next(model.classifier.parameters()) in model.optimizer.param_groups[1]["params"]
    )


def test_reset_or_keep_optimizer_and_scheduler(train_df, temp_model_dir):
    import copy
    from opensoundscape.utils import set_seed

    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    set_seed(0)
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )

    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=0,
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
        save_path=temp_model_dir,
        steps=0,
        batch_size=2,
        save_interval=10,
        num_workers=0,
        reset_optimizer=False,
        restart_scheduler=False,
    )

    # assert (model.optimizer.state_dict() == opt1.state_dict()).all()
    # check optimizer state dict equality
    assert str(model.optimizer.state_dict()) == str(opt1.state_dict())
    assert model.scheduler.state_dict()["last_epoch"] == 1
    assert model.scheduler.state_dict()["_step_count"] == 2

    # test that reset_optimizer=True, restart_scheduler=False
    # resets the optimizer to the initial state
    set_seed(0)
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=0,
        batch_size=2,
        save_interval=10,
        num_workers=0,
        reset_optimizer=True,
        restart_scheduler=True,
    )

    assert model.optimizer.state_dict()["state"] == {}
    assert model.scheduler.state_dict()["last_epoch"] == 0
    assert model.scheduler.state_dict()["_step_count"] == 1

    # No need to manually remove directory - fixture handles cleanup


def test_train_amp_cpu(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    # first test with cpu
    model.device = "cpu"
    model.use_amp = True
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_df)
    # No need to manually remove directory - fixture handles cleanup


def test_train_amp_cuda(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    # if cuda is available, test with cuda
    if torch.cuda.is_available():
        assert model.device.type == "cuda"
    else:
        return  # cannot test cuda
    model.use_amp = True
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_df)
    # No need to manually remove directory - fixture handles cleanup


def test_train_amp_mps(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    if torch.mps.is_available():
        assert model.device.type == "mps"
    else:
        return  # cannot test mps on this machine
    model.use_amp = True
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_df)
    # No need to manually remove directory - fixture handles cleanup


def test_train_resample_loss(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    cnn.use_resample_loss(model, train_df=train_df)
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    # No need to manually remove directory - fixture handles cleanup


def test_train_one_class(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    model.train(
        train_df[[0]],
        train_df[[0]],
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    # No need to manually remove directory - fixture handles cleanup


def test_single_target_setter():
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
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
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=1.0,
        sample_rate=22050,
        arch_weights=None,
    )
    model.single_target = True
    scores = model.predict(train_df_clips)
    assert len(scores) == 20
    model.eval(train_df_clips.values, scores.values)


def test_predict_on_list_of_files(test_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    scores = model.predict(test_df.index.values)
    assert len(scores) == 2


def test_predict_with_audio_root():
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    scores = model.predict(["silence_10s.mp3"], audio_root="tests/audio/")
    assert len(scores) == 2


def test_predict_and_embed_on_df_with_file_index(train_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    scores = model.predict(train_df)
    emb = model.embed(train_df)
    assert len(scores) == 4
    assert len(emb) == 4


def test_predict_on_empty_list():
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    scores = model.predict([])
    expected = ["file", "start_time", "end_time", 0, 1]
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
                    sample_rate=22050,
                    channels=4,
                )
                if arch_name in ("alexnet", "vgg11_bn"):
                    # don't use MPS bc of adaptive pooling implementation gap
                    # as of April 2026
                    if str(model.device) == "mps":
                        # model.device = "cpu"
                        continue  # skip test for MPS for these architectures because of adaptive pooling implementation gap as of April 2026
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
                    architecture=arch,
                    classes=[0, 1],
                    sample_duration=5.0,
                    sample_rate=22050,
                    channels=1,
                )
                if arch_name in ("alexnet", "vgg11_bn"):
                    # don't use MPS bc of adaptive pooling implementation gap
                    # as of April 2026
                    if str(model.device) == "mps":
                        # model.device = "cpu"
                        continue  # skip test for MPS for these architectures because of adaptive pooling implementation gap as of April 2026

            scores = model.predict(test_df.index.values)
            assert len(scores) == 2
        except Exception as e:
            raise Exception(f"{arch_name} failed") from e


def test_predict_on_clip_df(test_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=1.0,
        sample_rate=22050,
        arch_weights=None,
    )
    clip_df = make_clip_df(test_df.index.values[0:1], clip_duration=1.0)
    scores = model.predict(clip_df)
    assert len(scores) == 10


def test_prediction_overlap(test_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=None,
        arch_weights=None,
    )
    model.single_target = True
    scores = model.predict(test_df, overlap_fraction=0.5, final_clip=None)

    assert len(scores) == 3


def test_predict_on_one_file(test_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=10,
        sample_rate=22050,
        arch_weights=None,
    )
    p = test_df.index.values[0]
    scores = model.predict(p)
    assert len(scores) == 1
    scores = model.predict(Path(p))
    assert len(scores) == 1


def test_multi_target_prediction(train_df, test_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    scores = model.predict(test_df)

    assert len(scores) == 2


def test_predict_missing_file_is_invalid_sample(missing_file_df, test_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )

    with pytest.raises(FileNotFoundError):
        # first file is bad, will give ValueError
        model.predict(missing_file_df)

    good_then_bad_df = pd.concat([test_df.head(1), missing_file_df])

    scores, invalid_samples = model.predict(
        good_then_bad_df, return_invalid_samples=True
    )
    assert (
        len(scores) == 3
    )  # should have second row with nan values for the invalid sample
    isnan = lambda x: x != x
    assert np.all([isnan(score) for score in scores.iloc[2].values])
    assert len(invalid_samples) == 1

    # repeat with num_workers > 0 (issue #1180)
    scores, invalid_samples = model.predict(
        good_then_bad_df,
        return_invalid_samples=True,
        num_workers=1,
    )
    assert (
        len(scores) == 3
    )  # should have one row with nan values for the invalid sample
    assert np.all([isnan(score) for score in scores.iloc[2].values])
    # assert len(invalid_samples) == 1 # doesn't work with multiprocessing mode

    # repeat for embedding
    emb = model.embed(
        good_then_bad_df,
        num_workers=1,
    )
    assert len(emb) == 3  # should have one row with nan values for the invalid sample
    assert np.all([isnan(e) for e in emb.iloc[2].values])

    # and for embedding without avg pool
    # repeat for embedding
    emb = model.embed(
        good_then_bad_df,
        num_workers=1,
        avgpool=False,
    )
    assert (~np.isnan(emb[0:1, ::])).all()
    assert (np.isnan(emb[2, ::])).all()


def test_predict_wrong_input_error(test_df):
    """cannot pass a preprocessor or dataset to predict. only file paths as list or df"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    pre = SpectrogramPreprocessor(2.0, sample_rate=22050)
    with pytest.raises(AssertionError):
        model.predict(pre)
    with pytest.raises(AssertionError):
        ds = AudioFileDataset(test_df, pre)
        model.predict(ds)


def test_profile(train_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    profile = model.profile(
        train_df,
        batch_size=2,
        num_workers=0,
    )
    assert "preprocess_profile" in profile
    assert "preprocess_time_per_sample" in profile
    assert "backward_time_per_batch" in profile


def test_train_predict_inception(train_df, temp_model_dir):
    model = cnn.InceptionV3([0, 1], 5.0, weights=None, sample_rate=22050)
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )
    model.predict(train_df, num_workers=0)
    # No need to manually remove directory - fixture handles cleanup


def test_train_predict_architecture(train_df):
    """test passing architecture object to CNN class

    should internally update `channels` to match architecture (3 channels)
    """
    for num_channels in (1, 3):
        arch = resnet18(num_classes=2, weights=None, num_channels=num_channels)
        model = cnn.CNN(
            architecture=arch, classes=[0, 1], sample_duration=2, sample_rate=22050
        )
        model.predict(train_df, num_workers=0)
        assert model.preprocessor.channels == num_channels


def test_train_bad_index(train_df, temp_model_dir):
    """
    AssertionError catches case where index is not one of the allowed formats
    """
    model = cnn.CNN(
        "resnet18", [0, 1], sample_duration=2, sample_rate=22050, arch_weights=None
    )
    # reset the index so that train_df index is integers (not an allowed format)
    train_df = make_clip_df(train_df.index.values, clip_duration=2).reset_index()
    train_df[0] = np.random.choice([0, 1], size=10)
    train_df[1] = np.random.choice([0, 1], size=10)
    with pytest.raises(AssertionError):
        model.train(
            train_df,
            train_df,
            save_path=temp_model_dir,
            steps=1,
            batch_size=2,
            save_interval=10,
            num_workers=0,
        )


def test_predict_splitting_short_file(short_file_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scores = model.predict(short_file_df, final_clip=None)
        assert len(scores) == 0
        all_warnings = ""
        for wi in w:
            all_warnings += str(wi.message)
        assert "prediction_dataset" in all_warnings
        scores = model.predict(short_file_df)
        assert len(scores) == 1  # default mode now extend


def test_train_early_stopping(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    model.early_stopping_config.update(
        {"patience": 2, "min_delta": 0.01, "enabled": True}
    )
    assert model.early_stopping_config["enabled"] is True
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=20,
        batch_size=2,
        save_interval=-1,
        num_workers=0,
        validation_interval=1,
    )
    assert hasattr(model, "_best_score_early_stopping")
    assert model._best_step_early_stopping < 20


def test_train_revert_to_best_epoch(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=5,
        batch_size=2,
        save_interval=1,
        num_workers=0,
        validation_interval=1,
        reload_best_at_end=True,
    )
    # check that we reverted to best epoch at the end of training
    best_weights = torch.load(
        f"{temp_model_dir}/best.pickle", map_location=model.device, weights_only=False
    ).network.state_dict()
    assert are_state_dicts_close(model.network.state_dict(), best_weights)


def test_save_and_load_model(model_save_path):
    classes = [0, 1]
    cnn.CNN(
        architecture="alexnet",
        classes=classes,
        sample_duration=1.0,
        sample_rate=22050,
        arch_weights=None,
    ).save(model_save_path)
    m = cnn.load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == cnn.CNN

    cnn.InceptionV3(
        classes=classes, sample_duration=1.0, sample_rate=22050, weights=None
    ).save(model_save_path)
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
    m = cnn.CNN(
        architecture=arch, classes=classes, sample_duration=1.0, sample_rate=22050
    )
    m.save(model_save_path)
    m2 = cnn.load_model(model_save_path)
    assert type(m2.network) == type(arch)

    # remove the custom architecture from the ARCH_DICT when done
    del cnn_architectures.ARCH_DICT["my_alexnet_generator"]


def test_init_positional_args():
    cnn.CNN("resnet18", [0, 1], 0, 22050)


def test_save_load_and_train_model_resample_loss(
    train_df, model_save_path, temp_model_dir
):
    arch = resnet18(2, weights=None)
    classes = [0, 1]

    m = cnn.CNN(
        architecture=arch, classes=classes, sample_duration=1.0, sample_rate=22050
    )
    cnn.use_resample_loss(m, train_df)
    m.save(model_save_path, pickle=True)
    m2 = cnn.load_model(model_save_path)
    assert m2.classes == classes
    assert type(m2) == cnn.CNN
    assert isinstance(m2.loss_fn, ResampleLoss)
    # make sure it still trains ok after reloading w/resample loss
    m2.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )

    # No need to manually remove directory - fixture handles cleanup


def test_prediction_warns_different_classes(train_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=["a", "b"],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # raises warning bc test_df columns != model.classes
        model.predict(train_df)
        all_warnings = ""
        for wi in w:
            all_warnings += str(wi.message)
        assert "classes" in all_warnings

    # also for embed()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # raises warning bc test_df columns != model.classes
        model.embed(train_df)
        all_warnings = ""
        for wi in w:
            all_warnings += str(wi.message)
        assert "classes" in all_warnings


def test_train_raises_wrong_class_list(train_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=["different"],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    with pytest.raises(AssertionError):
        # raises AssertionError bc test_df columns != model.classes
        model.train(train_df, steps=1)


def test_train_raises_labels_outside_range(train_df):
    model = cnn.CNN(
        "resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    train_df.iat[0, 0] = 2
    with pytest.raises(AssertionError):
        # raises AssertionError bc values outside [0,1] not allowed
        model.train(train_df, steps=1)


def test_prediction_returns_consistent_values(train_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=["a", "b"],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    a = model.predict(train_df)
    b = model.predict(train_df)
    assert np.allclose(a.values, b.values, 1e-6)


def test_save_and_load_weights(model_save_path):
    arch = resnet18(2, weights=None, num_channels=1)
    model = cnn.CNN(
        architecture="resnet18",
        classes=["a", "b"],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    model.save_weights(model_save_path)
    model1 = cnn.CNN(arch, classes=["a", "b"], sample_duration=5.0, sample_rate=22050)
    model1.load_weights(model_save_path)
    assert np.array_equal(
        model.network.state_dict()["conv1.weight"].numpy(),
        model1.network.state_dict()["conv1.weight"].numpy(),
    )


def test_eval(train_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=2,
        sample_rate=22050,
        arch_weights=None,
    )
    scores = np.random.uniform(0, 1, (len(train_df), 2))
    model.eval(train_df.values, scores)


def test_eval_raises_bad_labels(train_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=2,
        sample_rate=22050,
        arch_weights=None,
    )
    scores = np.random.uniform(0, 1, (len(train_df), 2))
    train_df.iat[0, 0] = 2
    with pytest.raises(AssertionError):
        # raises AssertionError bc values outside [0,1] not allowed
        model.eval(train_df.values, scores)


def test_train_no_validation(train_df, temp_model_dir):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=2,
        sample_rate=22050,
        arch_weights=None,
    )
    model.train(train_df, save_path=temp_model_dir, steps=2)
    # No need to manually remove directory - fixture handles cleanup


def test_train_raise_errors(short_file_df, missing_file_df):
    files_df = pd.concat(
        [short_file_df, missing_file_df]
    )  # use 2 files. 1 file wrong is manually caught and userwarning raised
    files_df["class"] = [0, 1]  # add labels for training
    model = cnn.CNN(
        "resnet18",
        classes=["class"],
        sample_duration=2,
        sample_rate=22050,
        arch_weights=None,
    )
    with pytest.raises(PreprocessingError):
        model.train(files_df, raise_errors=True, steps=2)


def test_predict_raise_errors(short_file_df, onemin_wav_df):
    files_df = pd.concat(
        [short_file_df, onemin_wav_df]
    )  # use 2 files. 1 file wrong is manually caught and userwarning raised
    model = cnn.CNN(
        architecture="resnet18",
        classes=["class"],
        sample_duration=30,
        sample_rate=22050,
        arch_weights=None,
    )
    model.preprocessor.pipeline.bandpass.bypass = False  # ensure bandpass happens
    # add a bad param. this should be min_f
    model.preprocessor.pipeline.bandpass.params["low"] = 1

    with pytest.raises(PreprocessingError):
        model.predict(files_df, raise_errors=True)


def test_generate_cams(test_df):
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
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
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    samples = model.generate_samples(test_df)
    assert len(samples) == 2
    assert type(samples[0]) == AudioSample
    assert type(samples[0].data) == torch.Tensor
    assert type(samples[0].labels) == pd.Series


def test_generate_cams_batch_size(test_df):
    """smoke test for batch size > 1"""
    model = cnn.CNN(
        "resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    _ = model.generate_cams(test_df, batch_size=2)


def test_generate_cams_num_workers(test_df):
    """smoke test for num workers > 1"""
    model = cnn.CNN(
        "resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    _ = model.generate_cams(test_df, num_workers=2)


def test_generate_cams_scorecam_devices(test_df):
    """In pytorch_grad_cam <1.5.0 scorecam had device mismatch"""

    model = cnn.CNN(
        "resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    import pytorch_grad_cam

    _ = model.generate_cams(
        test_df,
        method=pytorch_grad_cam.ScoreCAM,
    )

    # very slow on cpu - but can uncomment to check
    # model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0, sample_rate=22050)
    # model.device = "cpu"
    # import pytorch_grad_cam

    # _ = model.generate_cams(
    #     test_df,
    #     method=pytorch_grad_cam.ScoreCAM,
    # )


def test_generate_cams_methods(test_df):
    """test each supported method both by passing class and string name"""

    model = cnn.CNN(
        "resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
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
                architecture=arch,
                classes=[0, 1],
                sample_duration=5.0,
                sample_rate=22050,
                channels=1,
            )
            if arch_name in ("alexnet", "vgg11_bn"):
                # don't use MPS bc of adaptive pooling implementation gap
                # as of April 2026
                if str(model.device) == "mps":
                    # model.device = "cpu"
                    continue  # skip test for MPS for these architectures because of adaptive pooling implementation gap as of April 2026
            _ = model.generate_cams(test_df.head(1))
        except Exception as e:
            raise Exception(f"{arch_name} failed") from e


def test_generate_cams_target_layers(test_df):
    """specify multiple target layers for cam"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    _ = model.generate_cams(
        test_df, target_layers=[model.network.layer3, model.network.layer4]
    )


def test_train_with_posixpath(train_df, temp_model_dir):
    """test that train works with pathlib.Path objects"""
    from pathlib import Path

    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )

    # turn the file paths into Path objects.
    posix_paths = [Path(p) for p in train_df.index]

    # change the index of train_df to be the Path objects
    train_df.index = posix_paths

    model.train(
        train_df,
        train_df,
        save_path=temp_model_dir,
        steps=1,
        batch_size=2,
        save_interval=10,
        num_workers=0,
    )

    # No need to manually remove directory - fixture handles cleanup


def test_predict_posixpath_missing_files(missing_file_df, test_df):
    """Test that predict works with pathlib.Path objects"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )

    missing_file_df.index = [Path(p) for p in missing_file_df.index]
    test_df.index = [Path(p) for p in test_df.index]
    with pytest.raises(FileNotFoundError):
        # if first sample's file not found, raises FileNotFoundError
        model.predict(missing_file_df)

    scores, invalid_samples = model.predict(
        pd.concat([test_df.head(1), missing_file_df]), return_invalid_samples=True
    )
    assert (
        len(scores) == 3
    )  # should have one row with nan values for the invalid sample
    isnan = lambda x: x != x
    assert np.all([isnan(score) for score in scores.iloc[2].values])
    assert len(invalid_samples) == 1
    assert (
        missing_file_df.index.values[0] in invalid_samples.reset_index()["file"].values
    )


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
                sample_rate=22050,
                arch_weights=None,
            )
            if arch in ("alexnet", "vgg11_bn"):
                # don't use MPS bc of adaptive pooling implementation gap
                # as of April 2026
                if str(m.device) == "mps":
                    m.device = "cpu"
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
        sample_rate=22050,
        arch_weights=None,
    )
    embeddings = m.embed(
        samples=test_df,
        avgpool=False,
        progress_bar=False,
        target_layer=m.network.layer4,
    )
    # was (2, 512, 7, 7) when spec shape = 224,224
    assert embeddings.shape == (2, 512, 9, 14)


def test_embed_return_array(test_df):
    # returns arrays rather than dataframes
    m = cnn.SpectrogramClassifier(
        classes=[0, 1],
        single_target=False,
        architecture="resnet18",
        sample_duration=5,
        sample_rate=22050,
        arch_weights=None,
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
        sample_rate=22050,
        arch_weights=None,
    )
    embeddings = m.embed(samples=train_df.head(1), avgpool=True, progress_bar=False)
    assert embeddings.shape == (1, 512)


def test_call_with_targets(test_df):
    """test that passing intermediate_layers to SpectrogramClassifier.__call__ returns tensors of expected shape"""
    model = cnn.SpectrogramClassifier(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
    )
    dl = model.predict_dataloader(test_df)
    outs = model(dl, targets=[model.network.layer1, model.network.layer4])
    assert len(outs.keys()) == 2
    assert np.shape(outs[model.network.layer1]) == (2, 64)
    assert np.shape(outs[model.network.layer4]) == (2, 512)
    outs = model(dl, targets=[model.network.layer4], avgpool_intermediates=False)
    assert np.shape(outs[model.network.layer4]) == (2, 512, 9, 14)  # (2, 512, 7, 7)


def test_batch_forward_returns_requested_targets():
    """Verify batch_forward() returns outputs for all requested layer targets.

    The batch_forward() method accepts a list of target layers and returns a dictionary
    with outputs from each target. This test verifies that intermediate layer outputs
    have correct shapes and that the special key -1 represents the final model output.
    Includes testing with and without average pooling applied to intermediate outputs.
    """
    model = cnn.SpectrogramClassifier(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        arch_weights=None,
        sample_rate=None,
    )
    model.device = "cpu"

    labels = pd.Series([1, 0], index=[0, 1], name=("tests/audio/silence_10s.mp3", 0, 5))
    sample1 = AudioSample(
        source="tests/audio/silence_10s.mp3",
        start_time=0,
        duration=5,
        labels=labels,
    )
    sample2 = AudioSample(
        source="tests/audio/silence_10s.mp3",
        start_time=5,
        duration=5,
        labels=labels,
    )
    sample1.data = torch.randn(1, 224, 224)
    sample2.data = torch.randn(1, 224, 224)
    sample1.is_alternative = False
    sample2.is_alternative = False

    outs = model.batch_forward(
        [sample1, sample2], targets=[-1, model.network.layer4], avgpool=False
    )
    assert set(outs.keys()) == {-1, model.network.layer4}
    assert np.shape(outs[-1]) == (2, 2)
    assert np.shape(outs[model.network.layer4]) == (2, 512, 7, 7)


def test_call_masks_invalid_alternative_samples():
    """Verify __call__() masks outputs to NaN for invalid/alternative samples.

    When the preprocessing pipeline fails on a sample, it returns a placeholder
    (alternative) sample. The model's __call__ method should detect these invalid
    samples (via is_alternative attribute) and mask their outputs to NaN, preventing
    them from being used in downstream processing or metrics calculations.
    """
    model = cnn.SpectrogramClassifier(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        arch_weights=None,
        sample_rate=None,
    )
    model.device = "cpu"

    labels = pd.Series([1, 0], index=[0, 1], name=("tests/audio/silence_10s.mp3", 0, 5))
    sample1 = AudioSample(
        source="tests/audio/silence_10s.mp3",
        start_time=0,
        duration=5,
        labels=labels,
    )
    sample2 = AudioSample(
        source="tests/audio/silence_10s.mp3",
        start_time=5,
        duration=5,
        labels=labels,
    )
    sample1.data = torch.randn(1, 224, 224)
    sample2.data = torch.randn(1, 224, 224)
    sample1.is_alternative = False
    sample2.is_alternative = True

    dataloader = torch.utils.data.DataLoader(
        [sample1, sample2], batch_size=2, collate_fn=identity
    )
    outs = model(dataloader, targets=[-1], progress_bar=False)
    assert np.isfinite(outs[-1][0]).all()
    assert np.isnan(outs[-1][1]).all()


def test_freeze_layers_except_and_unfreeze():
    model = cnn.SpectrogramClassifier(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=5.0,
        sample_rate=22050,
        arch_weights=None,
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
                architecture=arch,
                classes=[0, 1],
                sample_duration=5.0,
                sample_rate=22050,
                channels=1,
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


def test_change_classifier_all_arch():
    """change_classifier should change the classes attribute and the output layer of the network"""
    for arch_name in cnn_architectures.ARCH_DICT.keys():
        try:
            model = cnn.CNN(
                architecture=arch_name,
                classes=[0, 1],
                sample_duration=5.0,
                sample_rate=22050,
                channels=1,
            )
            if arch_name == "squeezenet1_0" or arch_name == "inception_v3":
                # not supported (squeezenet has conv2d classifier, inception has aux classifiers)
                continue
            else:
                new_clf = torch.nn.Linear(model.classifier.in_features, 3)
                model.change_classifier(new_clf, classes=[0, 1, 2])
                assert model.classes == [0, 1, 2]
                assert model.classifier.out_features == 3
                # try forward pass
                assert model.network(torch.randn(1, 1, 224, 224)).shape == (1, 3)

                # also try with MLPClassifier
                hidden_layers = ()
                model.change_classifier(
                    MLPClassifier(
                        input_size=model.classifier.in_features,
                        output_size=3,
                        hidden_layer_sizes=hidden_layers,
                        classes=[0, 1, 2],
                    ),
                    classes=[0, 1, 2],
                )
                assert model.classes == [0, 1, 2]
                assert model.classifier.out_features == 3
                # try forward pass
                assert model.network(torch.randn(1, 1, 224, 224)).shape == (1, 3)

        except Exception as e:
            raise Exception(f"{arch_name} failed") from e


def test_change_classes_all_arch():
    """change_classes should change the classes attribute and the output layer of the network"""
    for arch_name in cnn_architectures.ARCH_DICT.keys():
        try:
            model = cnn.CNN(
                architecture=arch_name,
                classes=[0, 1],
                sample_duration=5.0,
                sample_rate=22050,
                channels=1,
            )
            if arch_name == "squeezenet1_0" or arch_name == "inception_v3":
                # not supported (squeezenet has conv2d classifier, inception has aux classifiers)
                continue
            else:
                model.change_classes([0, 1, 2])
                assert model.classes == [0, 1, 2]
                assert model.classifier.out_features == 3
        except Exception as e:
            raise Exception(f"{arch_name} failed") from e


def test_change_classes_mlp_classifier():
    """Test change_classes method with MLPClassifier hidden layers"""
    # Create a CNN model with standard linear classifier
    model = cnn.CNN(
        architecture="resnet18",
        classes=["class_a", "class_b"],
        sample_duration=5.0,
        sample_rate=22050,
        channels=1,
        arch_weights=None,
    )

    # Store original classifier type and input features
    original_in_features = model.classifier.in_features
    assert isinstance(model.classifier, torch.nn.Linear)

    # Test 1: Change to MLPClassifier with single hidden layer
    new_classes = ["bird", "frog", "insect"]
    hidden_layers = [128]

    model.change_classes(new_classes, hidden_layers=hidden_layers)

    # Verify the classifier was replaced with MLPClassifier
    assert isinstance(model.classifier, MLPClassifier)
    assert model.classes == new_classes
    assert model.classifier.in_features == original_in_features
    assert model.classifier.out_features == len(new_classes)
    assert model.classifier.hidden_layer_sizes == tuple(hidden_layers)
    assert model.classifier.classes == new_classes

    # Test forward pass works
    dummy_input = torch.randn(
        2, 1, 224, 224
    )  # batch_size=2, channels=1, height=224, width=224
    output = model.network(dummy_input)
    assert output.shape == (2, len(new_classes))


def test_change_classes_mlp_multiple_hidden_layers():
    """Test change_classes with MLPClassifier having multiple hidden layers"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=["a", "b"],
        sample_duration=5.0,
        sample_rate=22050,
        channels=1,
        arch_weights=None,
    )

    original_in_features = model.classifier.in_features

    # Test with multiple hidden layers
    new_classes = ["dog", "cat", "bird", "other"]
    hidden_layers = [256, 128, 64]

    model.change_classes(new_classes, hidden_layers=hidden_layers)

    assert isinstance(model.classifier, MLPClassifier)
    assert model.classes == new_classes
    assert model.classifier.in_features == original_in_features
    assert model.classifier.out_features == len(new_classes)
    assert model.classifier.hidden_layer_sizes == tuple(hidden_layers)
    assert model.classifier.classes == new_classes

    # Test forward pass
    dummy_input = torch.randn(3, 1, 224, 224)
    output = model.network(dummy_input)
    assert output.shape == (3, len(new_classes))


def test_change_classes_mlp_no_hidden_layers():
    """Test change_classes with empty tuple for hidden layers (creates MLPClassifier with no hidden layers)"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=["x", "y"],
        sample_duration=5.0,
        sample_rate=22050,
        channels=1,
        arch_weights=None,
    )

    original_in_features = model.classifier.in_features

    # Test with empty tuple - should create MLPClassifier with no hidden layers
    new_classes = ["noise", "signal"]
    hidden_layers = ()

    model.change_classes(new_classes, hidden_layers=hidden_layers)

    assert isinstance(model.classifier, MLPClassifier)
    assert model.classes == new_classes
    assert model.classifier.in_features == original_in_features
    assert model.classifier.out_features == len(new_classes)
    assert model.classifier.hidden_layer_sizes == ()
    assert model.classifier.classes == new_classes

    # Test forward pass
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model.network(dummy_input)
    assert output.shape == (1, len(new_classes))


def test_change_classes_back_to_linear():
    """Test changing from MLPClassifier back to torch.nn.Linear"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=["a", "b"],
        sample_duration=5.0,
        sample_rate=22050,
        channels=1,
        arch_weights=None,
    )

    original_in_features = model.classifier.in_features

    # First change to MLPClassifier
    model.change_classes(["x", "y", "z"], hidden_layers=[64])
    assert isinstance(model.classifier, MLPClassifier)

    # Then change back to Linear (hidden_layers=None)
    new_classes = ["final_a", "final_b"]
    model.change_classes(new_classes, hidden_layers=None)

    assert isinstance(model.classifier, torch.nn.Linear)
    assert model.classes == new_classes
    assert model.classifier.in_features == original_in_features
    assert model.classifier.out_features == len(new_classes)

    # Test forward pass
    dummy_input = torch.randn(2, 1, 224, 224)
    output = model.network(dummy_input)
    assert output.shape == (2, len(new_classes))


def test_change_classes_mlp_from_existing_mlp():
    """Test changing classes when starting with MLPClassifier"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=["a", "b"],
        sample_duration=5.0,
        sample_rate=22050,
        channels=1,
        arch_weights=None,
    )

    # First create an MLPClassifier
    model.change_classes(["x", "y"], hidden_layers=[32])
    original_in_features = model.classifier.in_features
    assert isinstance(model.classifier, MLPClassifier)

    # Now change to a different MLPClassifier configuration
    new_classes = ["class1", "class2", "class3", "class4"]
    new_hidden_layers = [128, 64]

    model.change_classes(new_classes, hidden_layers=new_hidden_layers)

    assert isinstance(model.classifier, MLPClassifier)
    assert model.classes == new_classes
    assert model.classifier.in_features == original_in_features
    assert model.classifier.out_features == len(new_classes)
    assert model.classifier.hidden_layer_sizes == tuple(new_hidden_layers)
    assert model.classifier.classes == new_classes

    # Test forward pass
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model.network(dummy_input)
    assert output.shape == (1, len(new_classes))


def test_change_classes_invalid_hidden_layers():
    """Test change_classes with invalid hidden_layers parameter"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=["a", "b"],
        sample_duration=5.0,
        sample_rate=22050,
        channels=1,
        arch_weights=None,
    )

    # Test with invalid hidden_layers type
    with pytest.raises(ValueError, match="hidden_layers must be None"):
        model.change_classes(["x", "y"], hidden_layers="invalid")

    with pytest.raises(ValueError, match="hidden_layers must be None"):
        model.change_classes(["x", "y"], hidden_layers=123)


def test_change_classes_single_hidden_layer_list():
    """Test change_classes with single hidden layer specified as list"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=["a", "b"],
        sample_duration=5.0,
        sample_rate=22050,
        channels=1,
        arch_weights=None,
    )

    # Test with list instead of tuple
    new_classes = ["bird", "noise"]
    hidden_layers = [100]  # List instead of tuple

    model.change_classes(new_classes, hidden_layers=hidden_layers)

    assert isinstance(model.classifier, MLPClassifier)
    assert model.classifier.hidden_layer_sizes == (100,)  # Should be converted to tuple
    assert model.classifier.classes == list(new_classes)


def test_change_classes_preserves_device():
    """Test that change_classes preserves model device"""
    model = cnn.CNN(
        architecture="resnet18",
        classes=["a", "b"],
        sample_duration=5.0,
        sample_rate=22050,
        channels=1,
        arch_weights=None,
    )

    # Move model to CPU
    model.device = "cpu"  # setter converts to torch.device('cpu')

    # Change classes with MLPClassifier
    model.change_classes(["x", "y", "z"], hidden_layers=[64])

    # Check that new classifier is on the same device
    assert next(model.classifier.parameters()).device == torch.device("cpu")

    # Test forward pass on same device
    dummy_input = torch.randn(1, 1, 224, 224).to(model.device)
    output = model.network(dummy_input)
    assert output.device == model.device


def test_embed_to_hoplite_db_inserts_embeddings_and_commits(monkeypatch):
    """Verify embed_to_hoplite_db() inserts batch embeddings and commits progress to DB."""

    class FakeDataset(torch.utils.data.Dataset):
        def __init__(self, label_df):
            self.dataset = types.SimpleNamespace(label_df=label_df)
            self._samples = [
                types.SimpleNamespace(
                    source="tests/audio/silence_10s.mp3", start_time=0.0, duration=1.0
                ),
                types.SimpleNamespace(
                    source="tests/audio/silence_10s.mp3", start_time=1.0, duration=1.0
                ),
            ]

        def __len__(self):
            return len(self._samples)

        def __getitem__(self, idx):
            return self._samples[idx]

        def report(self, log=None):
            return pd.DataFrame(columns=["file", "start_time", "end_time", "error"])

    class FakeDB:
        def __init__(self):
            self.commits = 0
            self._deployments = []

        def get_all_deployments(self, filter=None):
            return self._deployments

        def insert_deployment(self, name, project=""):
            self._deployments.append(
                types.SimpleNamespace(id=42, name=name, project=project)
            )
            return 42

        def get_all_recordings(self, filter=None):
            return []

        def commit(self):
            self.commits += 1

    inserted_batches = []

    def fake_insert_embeddings(
        db,
        batch_samples,
        batch_embeddings,
        overflow_mode,
        file_to_id,
        file_to_datetime=None,
        audio_root=None,
        deployment_id=None,
    ):
        inserted_batches.append(
            (len(batch_samples), batch_embeddings.shape[0], deployment_id)
        )
        return []

    def fake_load_or_create(db, embedding_dim=None, cfg=None):
        return db

    def fake_handle_existing_windows(
        db,
        clips,
        embedding_exists_mode,
        deployment_id=None,
        deployment_name=None,
        project=None,
        rounding_precision=6,
    ):
        return None

    monkeypatch.setattr(cnn, "_require_hoplite", lambda: None)
    monkeypatch.setattr(cnn, "_check_or_set_model_id", lambda db, model_id: None)

    import opensoundscape.vector_database as vector_db

    monkeypatch.setattr(
        vector_db, "load_or_create_hoplite_usearch_db", fake_load_or_create
    )
    monkeypatch.setattr(
        vector_db, "_handle_existing_windows", fake_handle_existing_windows
    )
    monkeypatch.setattr(vector_db, "_insert_embeddings", fake_insert_embeddings)

    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=1.0,
        sample_rate=22050,
        arch_weights=None,
    )
    label_df = pd.DataFrame(
        {0: [1, 0], 1: [0, 1]},
        index=pd.MultiIndex.from_tuples(
            [
                ("tests/audio/silence_10s.mp3", 0.0, 1.0),
                ("tests/audio/silence_10s.mp3", 1.0, 2.0),
            ],
            names=["file", "start_time", "end_time"],
        ),
    )

    model.predict_dataloader = lambda *args, **kwargs: torch.utils.data.DataLoader(
        FakeDataset(label_df), batch_size=1, collate_fn=identity
    )
    model._check_or_get_default_embedding_layer = lambda target_layer=None: "fake_layer"
    model.batch_forward = lambda batch_samples, targets, avgpool=True: {
        "fake_layer": np.ones((len(batch_samples), 2), dtype=np.float32)
    }

    db = FakeDB()
    out_db, info = model.embed_to_hoplite_db(
        samples=label_df,
        db=db,
        deployment="dep-1",
        batch_size=1,
        commit_frequency_batches=1,
        progress_bar=False,
    )

    assert out_db is db
    assert len(inserted_batches) == 2
    assert all(item[0] == 1 for item in inserted_batches)
    assert db.commits >= 2
    assert "insertion_failures" in info


def test_similarity_search_hoplite_db_returns_compiled_results(monkeypatch):
    """Verify similarity_search_hoplite_db() embeds queries and wraps vector-db search results."""
    monkeypatch.setattr(cnn, "_require_hoplite", lambda: None)

    model = cnn.CNN(
        architecture="resnet18",
        classes=[0, 1],
        sample_duration=1.0,
        arch_weights=None,
        sample_rate=22050,
    )

    emb_df = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=pd.MultiIndex.from_tuples(
            [("q1.wav", 0.0, 1.0), ("q2.wav", 1.0, 2.0)],
            names=["file", "start_time", "end_time"],
        ),
    )
    model.embed = lambda query_samples, audio_root=None, **kwargs: emb_df

    import opensoundscape.vector_database as vector_db

    monkeypatch.setattr(
        vector_db,
        "similarity_search_hoplite_db",
        lambda **kwargs: [{"file": "m.wav", "window_id": 9, "sort_score": 0.9}],
    )

    results = model.similarity_search_hoplite_db(
        query_samples=["q1.wav", "q2.wav"],
        db=object(),
        num_results=1,
        audio_root="/audio",
    )

    assert len(results) == 2
    assert results[0]["query"]["file"] == "q1.wav"
    assert results[0]["query"]["audio_root"] == "/audio"
    assert results[0]["results"][0]["window_id"] == 9


@onnx_deps
def test_save_onnx(onnx_save_path):
    from opensoundscape import CNN, preprocessors

    model = CNN(
        architecture="efficientnet_b0",
        classes=[0, 1, 2, 3],
        sample_duration=3,
        preprocessor_cls=preprocessors.TorchSpectrogramPreprocessor,
        sample_rate=32000,
        arch_weights=None,
    )
    onnx_program = model.save_onnx(onnx_save_path)

    # Using the saved model for inference with onnx runtime:

    import onnx, onnxruntime
    import numpy as np

    combined_model = onnx.load(onnx_save_path)
    output_names = [node.name for node in combined_model.graph.output]

    onnx.checker.check_model(combined_model)

    EP_list = [
        "CPUExecutionProvider"
    ]  # ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(onnx_save_path, providers=EP_list)

    # make up some random inputs
    audio_samples_per_input = (
        combined_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value
    )
    batch_size = 3
    input_batched = np.random.rand(batch_size, 1, audio_samples_per_input).astype(
        np.float32
    )

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: input_batched}
    ort_outs = ort_session.run(None, ort_inputs)

    # restore the name-value dictionary mapping of outputs
    outs_dict = {name: ort_outs[i] for i, name in enumerate(output_names)}
    assert outs_dict["classifier"].shape == (batch_size, 4)
    assert outs_dict["embedding"].shape == (batch_size, 1280)
    assert outs_dict["sample"].shape == (batch_size, 1, 257, 376)

    # Example 2: Exporting a model with customized preprocessing transforms
    model = CNN(
        architecture="efficientnet_b0",
        classes=[0, 1, 2, 3],
        sample_duration=3,
        preprocessor_cls=preprocessors.TorchSpectrogramPreprocessor,
        sample_rate=32000,
        bandpass_range=(3000, 10000),
        lower_dB_range=-30,
        rescale_mean_sd=(-30, 20),
        spec_nfft=512,
        spec_window_length=512,
        spec_hop_length=128,
        # resize_ft=(200, 512), # using resize_ft breaks serialization for json save/load!
        n_mels=64,
    )
    onnx_program = model.save_onnx(onnx_save_path)

    # Example 3: Writing a custom list of preprocessing transforms
    import torchaudio
    from opensoundscape import CNN, preprocessors

    model = CNN(
        "resnet18", classes=[0], sample_duration=5, arch_weights=None, sample_rate=32000
    )
    # custom list of torchaudio and torchvision transforms
    my_transforms = [
        torchaudio.transforms.Spectrogram(
            n_fft=512,
            win_length=512,
            hop_length=128,
            center=False,
        ),
        torchaudio.transforms.AmplitudeToDB(top_db=80),
    ]
    model.preprocessor = preprocessors.TorchSpectrogramPreprocessor(
        sample_rate=32000,
        sample_duration=model.preprocessor.sample_duration,
        torch_transforms=my_transforms,
    )
    onnx_program = model.save_onnx(onnx_save_path)
