from opensoundscape.torch.models.utils import tensor_binary_predictions

import pytest
import torch


def test_tensor_binary_predictions():
    scores = torch.stack((torch.arange(-5, 5, 1), torch.arange(-10, 0, 1)))
    assert (
        torch.sum(tensor_binary_predictions(scores, mode="multi_target", threshold=0))
        == 20
    )
    assert (
        torch.sum(tensor_binary_predictions(scores, mode="multi_target", threshold=1.1))
        == 0
    )
    assert (
        torch.sum(
            tensor_binary_predictions(scores, mode="multi_target", threshold=[1, 0] * 5)
        )
        == 10
    )
    with pytest.raises(ValueError):
        tensor_binary_predictions(scores, mode="wrong", threshold=0.5)
    with pytest.raises(ValueError):
        tensor_binary_predictions(scores, mode="multi_target", threshold=[0.5, 0])
