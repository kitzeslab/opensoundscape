#!/usr/bin/env python3
import pytest
from opensoundscape.metrics import Metrics
import numpy as np
from random import choice
from sklearn.metrics import confusion_matrix


@pytest.fixture()
def basic_confusion_matrix():
    confusion_matrix = np.array([[7, 8, 9], [1, 2, 3], [3, 2, 1]])
    metrics = Metrics([0, 1, 2], confusion_matrix.sum())
    metrics.confusion_matrix = confusion_matrix
    return metrics


def test_confusion_matrix_agrees_with_sklearn():
    choices = list(range(10))
    observations = 100
    metrics = Metrics(choices, observations)
    targets = [None] * observations
    predictions = [None] * observations
    for idx in range(observations):
        targets[idx] = choice(choices)
        predictions[idx] = choice(choices)
        metrics.accumulate_batch_metrics(0, [targets[idx]], [predictions[idx]])
    sklearn_confusion_matrix = confusion_matrix(targets, predictions, labels=choices)
    np.testing.assert_array_equal(sklearn_confusion_matrix, metrics.confusion_matrix)


def test_confusion_matrix_returns_correct_values(basic_confusion_matrix):
    # Approximate results
    precisions = np.array([0.636, 0.167, 0.077])
    recalls = np.array([0.292, 0.333, 0.167])
    f1s = np.array([0.4, 0.222, 0.105])

    metrics_d = basic_confusion_matrix.compute_epoch_metrics()

    np.testing.assert_array_almost_equal(precisions, metrics_d["precision"], decimal=3)
    np.testing.assert_array_almost_equal(recalls, metrics_d["recall"], decimal=3)
    np.testing.assert_array_almost_equal(f1s, metrics_d["f1"], decimal=3)
