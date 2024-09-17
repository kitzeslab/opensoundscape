from opensoundscape import sample
import pandas as pd
import math
import torch


def test_audio_sample():
    sample.AudioSample("path")


def test_audio_sample_from_series():
    series = pd.Series(name=("path", 0, 1), index=["a", "b"], data=[0, 1])
    s = sample.AudioSample.from_series(series)
    assert s.labels["a"] == 0 and s.labels["b"] == 1


def test_audio_series_returns_copy():
    series = pd.Series(name=("path", 0, 1), index=["a", "b"], data=[0, 1])
    s = sample.AudioSample.from_series(series)
    s.labels["a"] = 1
    assert s.labels["a"] == 1 and series["a"] == 0


def test_audio_sample_categorical_labels():
    series = pd.Series(name=("path", 0, 1), index=["a", "b"], data=[0, 1])
    s = sample.AudioSample.from_series(series)
    assert s.categorical_labels == ["b"]


def test_audio_sample_end_time():
    series = pd.Series(name=("path", 2, 5), index=["a", "b"], data=[0, 1])
    s = sample.AudioSample("path", start_time=3, duration=2)
    assert math.isclose(s.end_time, 5, abs_tol=1e-8)


def test_collate_samples():
    """collate should return tensors of joined data and joined labels"""
    l = pd.Series(name=("path", 2, 5), index=["a"], data=[0])
    s = sample.AudioSample(torch.Tensor([[1, 2, 1], [0, 2, 3]]), labels=l)
    collated = sample.collate_audio_samples_to_dict([s, s, s, s])
    assert list(collated["samples"].shape) == [4, 2, 3]
    assert list(collated["labels"].shape) == [4, 1]
