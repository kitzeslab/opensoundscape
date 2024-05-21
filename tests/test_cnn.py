import pandas as pd
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import shutil

import warnings

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


def test_predict_on_empty_list():
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
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
                num_classes=2, num_channels=1, weights=None
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
    clip_df = make_clip_df(test_df.index.values[0:1], clip_duration=1.0)
    scores = model.predict(clip_df)
    assert len(scores) == 10


def test_prediction_overlap(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    model.single_target = True
    scores = model.predict(test_df, overlap_fraction=0.5)

    assert len(scores) == 3


def test_predict_on_one_file(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=10)
    p = test_df.index.values[0]
    scores = model.predict(p)
    assert len(scores) == 1
    scores = model.predict(Path(p))
    assert len(scores) == 1


def test_multi_target_prediction(train_df, test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    scores = model.predict(test_df)

    assert len(scores) == 2


def test_predict_missing_file_is_invalid_sample(missing_file_df, test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)

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
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
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
    """test passing architecture object to CNN class"""
    arch = alexnet(2, weights=None)
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
    train_df = make_clip_df(train_df.index.values, clip_duration=2)
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
        all_warnings = ""
        for wi in w:
            all_warnings += str(wi.message)
        assert "prediction_dataset" in all_warnings


def test_save_and_load_model(model_save_path):
    arch = alexnet(2, weights=None)
    classes = [0, 1]

    cnn.CNN(arch, classes, 1.0).save(model_save_path)
    m = cnn.load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == cnn.CNN

    cnn.InceptionV3(classes, 1.0, weights=None).save(model_save_path)
    m = cnn.load_model(model_save_path)
    assert m.classes == classes
    assert type(m) == cnn.InceptionV3


def test_save_and_load_torch_dict(model_save_path):
    arch = alexnet(2, weights=None)
    classes = [0, 1]
    with pytest.warns(UserWarning):
        # warns user bc can't recreate custom architecture
        cnn.CNN(arch, classes, 1.0).save_torch_dict(model_save_path)
        # can do model.architecture_name='alexnet' to enable reloading

    # works when arch is string
    cnn.CNN("resnet18", classes, 1.0).save_torch_dict(model_save_path)
    m = cnn.CNN.from_torch_dict(model_save_path)
    assert m.classes == classes
    assert type(m) == cnn.CNN

    # not implemented for InceptionV3 (from_torch_dict raises NotImplementedError)


def test_save_load_and_train_model_resample_loss(train_df):
    arch = alexnet(2, weights=None)
    classes = [0, 1]

    m = cnn.CNN(arch, classes, 1.0)
    cnn.use_resample_loss(m, train_df)
    m.save("tests/models/saved1.model")
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
    arch = resnet18(2, weights=None)
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
    model = cnn.CNN("resnet18", classes=["class"], sample_duration=30)
    model.preprocessor.pipeline.bandpass.bypass = False  # ensure bandpass happens
    model.preprocessor.pipeline.bandpass.params[
        "low"
    ] = 1  # add a bad param. this should be min_f

    with pytest.raises(PreprocessingError):
        model.predict(files_df, raise_errors=True)


def test_generate_cams(test_df):
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    samples = model.generate_cams(test_df)
    assert len(samples) == 2
    assert type(samples[0]) == AudioSample
    assert type(samples[0].cam) == CAM
    assert type(samples[0].cam.activation_maps) == pd.Series
    assert samples[0].cam.gbp_maps is None

    samples = model.generate_cams(test_df, guided_backprop=True, method=None)
    assert type(samples[0].cam.gbp_maps) == pd.Series
    assert samples[0].cam.activation_maps is None


def test_generate_cams_batch_size(test_df):
    """smoke test for batch size > 1"""
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    _ = model.generate_cams(test_df, batch_size=2)


def test_generate_cams_num_workers(test_df):
    # gives error about pickling #TODO
    """smoke test for num workers > 1"""
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    _ = model.generate_cams(test_df, num_workers=2)


def test_generate_cams_methods(test_df):
    """test each supported method both by passing class and string name"""

    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    import pytorch_grad_cam

    methods_dict = {
        "gradcam": pytorch_grad_cam.GradCAM,
        "hirescam": pytorch_grad_cam.HiResCAM,
        "scorecam": opensoundscape.ml.utils.ScoreCAM,  # pytorch_grad_cam.ScoreCAM,
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


# TODO: should we test the default target layer doesn't cause errors for each architecture in cnn_architectures?


def test_generate_cams_target_layers(test_df):
    """specify multiple target layers for cam"""
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    _ = model.generate_cams(
        test_df, target_layers=[model.network.layer3, model.network.layer4]
    )


def test_train_with_posixpath(train_df):
    """test that train works with pathlib.Path objects"""
    from pathlib import Path

    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)

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
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)

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
    model = cnn.CNN("resnet18", classes=[0, 1], sample_duration=5.0)
    with pytest.warns(DeprecationWarning):
        scores = model.predict(test_df, overlap_fraction=0.5)
        assert len(scores) == 3
    with pytest.raises(AssertionError):
        model.predict(test_df, overlap_fraction=0.5, clip_overlap_fraction=0.5)
