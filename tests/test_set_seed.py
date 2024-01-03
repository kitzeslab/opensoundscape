from opensoundscape.utils import set_seed
import numpy as np
import torch
import random
from opensoundscape.ml import cnn, cnn_architectures

import pytest

pytestmark = pytest.mark.parametrize("input", [1, 11, 13, 42, 59, 666, 1234])


def test_torch_rand(input):
    set_seed(input)
    tr1 = torch.rand(100)

    set_seed(input)
    tr2 = torch.rand(100)

    set_seed(input + 1)
    tr3 = torch.rand(100)

    assert all(tr1 == tr2) & any(tr1 != tr3)


def test_numpy_random_rand(input):
    set_seed(input)
    nr1 = np.random.rand(100)

    set_seed(input)
    nr2 = np.random.rand(100)

    set_seed(input + 1)
    nr3 = np.random.rand(100)

    assert all(nr1 == nr2) & any(nr1 != nr3)


def test_radom_sample(input):
    list1000 = list(range(1, 1000))

    set_seed(input)
    rs1 = random.sample(list1000, 100)

    set_seed(input)
    rs2 = random.sample(list1000, 100)

    set_seed(input + 1)
    rs3 = random.sample(list1000, 100)

    assert (rs1 == rs2) & (rs1 != rs3)


def test_cnn(input):
    set_seed(input)
    model_resnet1 = cnn_architectures.resnet18(num_classes=10, weights=None)
    lw1 = model_resnet1.layer1[0].conv1.weight

    set_seed(input)
    model_resnet2 = cnn_architectures.resnet18(num_classes=10, weights=None)
    lw2 = model_resnet2.layer1[0].conv1.weight

    set_seed(input + 1)
    model_resnet3 = cnn_architectures.resnet18(num_classes=10, weights=None)
    lw3 = model_resnet3.layer1[0].conv1.weight

    assert torch.all(lw1 == lw2) & torch.any(lw1 != lw3)
