#!/usr/bin/env python3
import pytest
import numpy as np
import torch
import pandas as pd

from opensoundscape import metrics


# metrics are currently implicitly tested through test_cnn
# however, we should add explicit tests
def test_multitarget_metrics():
    x = [[0, 1], [0.4, 1], [0.6, 1], [1, 0]]
    y = [[0, 1], [1, 0], [0, 1], [1, 0]]
    out = metrics.multi_target_metrics(y, x, [0, 1], 0.5)
    assert np.isclose(out["precision"], 0.58333333, 1e-5)
    assert np.isclose(out["recall"], 0.75, 1e-5)
    assert np.isclose(out["map"], 0.75, 1e-5)
    assert np.isclose(out["au_roc"], 0.75, 1e-5)
    assert np.isclose(out[0]["avg_precision"], 0.8333333333333333, 1e-5)
    assert np.isclose(out[0]["au_roc"], 0.75, 1e-5)


def test_singletarget_metrics():
    x = [[0, 1], [1, 0], [1, 0]]
    y = [[0, 1], [0, 1], [1, 0]]
    out = metrics.single_target_metrics(y, x)
    assert out["precision"] == 1
    assert out["recall"] == 0.5


def test_predict_multi_target_labels():
    scores = torch.stack((torch.arange(-5, 5, 1), torch.arange(-10, 0, 1)))
    assert torch.sum(metrics.predict_multi_target_labels(scores, threshold=0)) == 5
    assert torch.sum(metrics.predict_multi_target_labels(scores, threshold=1.1)) == 3
    assert (
        torch.sum(metrics.predict_multi_target_labels(scores, threshold=[1, 0] * 5))
        == 5
    )
    with pytest.raises(ValueError):
        metrics.predict_multi_target_labels(scores, threshold="string")
    with pytest.raises(AssertionError):
        # must raise ValueError because threshold is length 2, but there are 3 classes
        metrics.predict_multi_target_labels(scores, threshold=[0.5, 0])

    # should return input type: list, numpy, pandas, torch
    scores = [[0.3, 0.2], [0.5, 0.1]]
    assert isinstance(metrics.predict_multi_target_labels(scores, 0.5), list)
    scores = np.array(scores)
    assert isinstance(metrics.predict_multi_target_labels(scores, 0.5), np.ndarray)
    scores = torch.Tensor(scores)
    assert isinstance(metrics.predict_multi_target_labels(scores, 0.5), torch.Tensor)
    scores = pd.DataFrame(scores.numpy())
    assert isinstance(metrics.predict_multi_target_labels(scores, 0.5), pd.DataFrame)


def test_predict_single_target_labels():
    scores = [[0.2, 0.3], [0.9, 0.4]]
    assert np.allclose(
        metrics.predict_single_target_labels(scores), np.array([[0, 1], [1, 0]])
    )

    with pytest.raises(TypeError):  # does not take threshold argument
        metrics.predict_single_target_labels(scores, threshold=0.5)

    # should return input type: list, numpy, pandas, torch
    scores = [[0.3, 0.2], [0.5, 0.1]]
    assert isinstance(metrics.predict_single_target_labels(scores), list)
    scores = np.array(scores)
    assert isinstance(metrics.predict_single_target_labels(scores), np.ndarray)
    scores = torch.Tensor(scores)
    assert isinstance(metrics.predict_single_target_labels(scores), torch.Tensor)
    scores = pd.DataFrame(scores.numpy())
    assert isinstance(metrics.predict_single_target_labels(scores), pd.DataFrame)
