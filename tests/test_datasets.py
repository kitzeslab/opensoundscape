import pytest
import numpy as np
import pandas as pd
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.preprocess.utils import PreprocessingError
import warnings
from opensoundscape.ml.datasets import AudioFileDataset, AudioSplittingDataset


@pytest.fixture()
def dataset_df():
    paths = ["tests/audio/silence_10s.mp3", "tests/audio/silence_10s.mp3"]
    labels = [[1, 0], [0, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def short_file_df():
    paths = ["tests/audio/veryshort.wav"]
    labels = [[0, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def bad_good_df():
    paths = ["tests/audio/veryshort.wav", "tests/audio/silence_10s.mp3"]
    labels = [[1, 0], [1, 0]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def overlay_df():
    paths = ["tests/audio/rugr_drum.wav"]
    labels = [[0, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def overlay_df_all_positive():
    paths = ["tests/audio/rugr_drum.wav"]
    labels = [[1, 1]]
    return pd.DataFrame(index=paths, data=labels, columns=[0, 1])


@pytest.fixture()
def pre():
    return SpectrogramPreprocessor(sample_duration=2.0)


@pytest.fixture()
def overlay_pre(overlay_df):
    return SpectrogramPreprocessor(sample_duration=2.0, overlay_df=overlay_df)


@pytest.fixture()
def small_dataset(dataset_df, overlay_pre):
    return AudioFileDataset(dataset_df, overlay_pre)


def test_subset_dataset(small_dataset):
    small_dataset.sample(n=1)
    small_dataset.sample(frac=1)
    small_dataset.head(1)


def test_audio_file_dataset(dataset_df, pre):
    """should return tensor and labels"""
    pre.bypass_augmentation = False
    pre.height = 224
    pre.width = 224
    pre.channels = 3
    dataset = AudioFileDataset(dataset_df, pre)
    sample1 = dataset[0]
    assert sample1.data.numpy().shape == (3, 224, 224)
    assert dataset[0].labels.values.shape == (2,)


def test_audio_file_dataset_no_reshape(dataset_df, pre):
    """should return tensor and labels. Tensor is the same as the shape of the spectrogram"""
    pre.bypass_augmentation = False
    dataset = AudioFileDataset(dataset_df, pre)
    sample1 = dataset[0]
    # should be the same shape as
    # Spectrogram.from_audio(Audio.from_file('tests/audio/silence_10s.mp3',duration=2)).bandpass(0,11025)
    assert sample1.data.numpy().shape == (1, 129, 343)


def test_spec_preprocessor_augment_off(dataset_df, pre):
    """should return same image each time"""
    dataset = AudioFileDataset(dataset_df, pre, bypass_augmentations=True)
    sample1 = dataset[0].data.numpy()
    sample2 = dataset[0].data.numpy()
    assert np.array_equal(sample1, sample2)


def test_spec_preprocessor_augment_on(dataset_df, pre):
    """should return different images each time"""
    dataset = AudioFileDataset(dataset_df, pre)
    sample1 = dataset[0].data
    sample2 = dataset[0].data
    assert not np.array_equal(sample1, sample2)


def test_spec_preprocessor_overlay(dataset_df, overlay_pre):
    dataset = AudioFileDataset(dataset_df, overlay_pre)
    sample1 = dataset[0].data
    dataset.preprocessor.pipeline.overlay.bypass = True
    sample2 = dataset[0].data
    assert not np.array_equal(sample1, sample2)


def test_overlay_tries_different_sample(dataset_df, bad_good_df):
    pre = SpectrogramPreprocessor(sample_duration=2.0, overlay_df=bad_good_df)
    dataset = AudioFileDataset(dataset_df, pre)
    # should try to load the bad sample, then load the good one
    dataset[0].data


def test_overlay_different_class(dataset_df, overlay_pre):
    """just make sure it runs and doesn't hang"""
    overlay_pre.pipeline.overlay.set(overlay_class="different")
    dataset = AudioFileDataset(dataset_df, overlay_pre)
    dataset[0].data


def test_overlay_no_valid_samples(dataset_df, overlay_df_all_positive):
    pre = SpectrogramPreprocessor(
        sample_duration=2.0, overlay_df=overlay_df_all_positive
    )
    dataset = AudioFileDataset(dataset_df, pre)
    dataset.preprocessor.pipeline.overlay.set(overlay_class="different")
    with pytest.raises(PreprocessingError):
        sample1 = dataset[0]  # no samples with "different" labels


def test_overlay_specific_class(dataset_df, overlay_pre):
    """just make sure it runs and doesn't hang"""
    dataset = AudioFileDataset(dataset_df, overlay_pre)
    dataset.preprocessor.pipeline.overlay.set(overlay_class=1)
    dataset[0]


def test_overlay_with_weight_range(dataset_df, overlay_pre):
    """overlay should allow range [min,max] for overlay_weight"""
    dataset = AudioFileDataset(dataset_df, overlay_pre)
    dataset.preprocessor.pipeline.overlay.set(overlay_weight=[0.3, 0.7])
    dataset[0]


def test_overlay_with_invalid_weight_range(dataset_df, overlay_pre):
    """overlay should allow range [min,max] for overlay_weight"""
    dataset = AudioFileDataset(dataset_df, overlay_pre)
    with pytest.raises(PreprocessingError):
        dataset.preprocessor.pipeline.overlay.set(overlay_weight=[0.1, 1.1])
        sample1 = dataset[0]
    with pytest.raises(PreprocessingError):
        dataset.preprocessor.pipeline.overlay.set(overlay_weight=[0.1, 0.5, 0.9])
        dataset[0]


def test_overlay_update_labels(dataset_df, overlay_pre):
    """should return different images each time"""
    dataset = AudioFileDataset(dataset_df, overlay_pre)

    dataset.preprocessor.pipeline.overlay.set(overlay_class="different")
    dataset.preprocessor.pipeline.overlay.set(update_labels=True)
    sample = dataset[0]
    assert np.array_equal(sample.labels.values, [1, 1])


def test_overlay_update_labels_duplicated_index(dataset_df, overlay_df):
    """duplicate indices of overlay_df are now removed, resolving
    a bug that caused duplicated indices to return 2-d labels.
    """
    overlay_df = pd.concat([overlay_df, overlay_df])
    overlay_pre = SpectrogramPreprocessor(2.0, overlay_df=overlay_df)
    dataset = AudioFileDataset(dataset_df, overlay_pre)
    dataset.preprocessor.pipeline.overlay.set(overlay_class="different")
    dataset.preprocessor.pipeline.overlay.set(update_labels=True)
    sample = dataset[0]
    assert np.array_equal(sample.labels.values, [1, 1])


def test_overlay_criterion_fn(dataset_df, overlay_pre):
    """should only return a different overlay if
    criterion_fn returns True
    """
    dataset = AudioFileDataset(dataset_df, overlay_pre)
    # dataset.preprocessor.pipeline.overlay.set(overlay_class="different")
    dataset.preprocessor.pipeline.overlay.set(
        criterion_fn=lambda x: x.labels.values[0] == 1
    )

    # trick it into only performing the overlay, no other augmentations
    # by marking overlay as not an augmentation
    dataset.bypass_augmentations = True
    dataset.preprocessor.pipeline.overlay.is_augmentation = False

    sample1 = dataset[0]  # labels are [1,0], gets overlay
    sample2 = dataset[1]  # labels are [0,1], no overlay

    # now, turn off overlay
    dataset.preprocessor.pipeline.overlay.bypass = True
    sample1_noaug = dataset[0]
    sample2_noaug = dataset[1]

    # this one should have been augmented, and be different from no augmentation
    assert not np.array_equal(sample1.data, sample1_noaug.data)
    # this one should be the same with and without augmentation
    assert np.array_equal(sample2.data, sample2_noaug.data)


def test_audio_splitting_dataset(dataset_df, pre):
    dataset = AudioSplittingDataset(dataset_df, pre)
    assert len(dataset) == 10

    # load a sample
    dataset[0]


def test_audio_splitting_dataset_overlap(dataset_df, pre):
    dataset = AudioSplittingDataset(dataset_df, pre, overlap_fraction=0.5)
    assert len(dataset) == 18

    # load a sample
    dataset[17]
