import numpy as np
import pytest
import pandas as pd
import pytz
import datetime

from opensoundscape.ml import utils as ml_utils
from opensoundscape import AudioSample
import torch


def test_apply_activation_layer():
    x = torch.tensor([[1, 2, 3]])
    y = ml_utils.apply_activation_layer(x, "sigmoid")
    assert np.allclose(y, torch.tensor([[0.7311, 0.8808, 0.9526]]), atol=1e-4)

    y = ml_utils.apply_activation_layer(x, "softmax")
    assert np.allclose(y, torch.tensor([[0.0900, 0.2447, 0.6652]]), atol=1e-4)


def test_collate_audio_samples_to_tensors():
    data = torch.tensor([[1, 2, 3], [4, 5, 6]])
    s = AudioSample(data, labels=torch.tensor([1, 0, 0]))
    batch = [s, s, s, s]
    batched_data, batched_labels = ml_utils.collate_audio_samples_to_tensors(batch)
    assert batched_data.shape == (4, 2, 3)
    assert batched_labels.shape == (4, 3)
    assert type(batched_data) == torch.Tensor
    assert type(batched_labels) == torch.Tensor
