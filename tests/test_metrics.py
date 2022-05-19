#!/usr/bin/env python3
import pytest
import numpy as np

from opensoundscape import metrics

# metrics are currently implicitly tested through test_cnn
# however, we should add explicit tests
def test_multitarget_metrics():
    x = [[0, 1], [1, 0]]
    y = [[0, 1], [1, 1]]
    out = metrics.multi_target_metrics(y, x, [0, 1], 0.5)
    assert out["precision"] == 1
    assert out["recall"] == 0.75


def test_singletarget_metrics():
    x = [[0, 1], [1, 0], [1, 0]]
    y = [[0, 1], [0, 1], [1, 0]]
    out = metrics.single_target_metrics(y, x, [0, 1])
    assert out["precision"] == 1
    assert out["recall"] == 0.5
