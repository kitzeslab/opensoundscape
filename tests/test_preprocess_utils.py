from opensoundscape.preprocess import utils
import pytest
import torch
import inspect


def test_get_args():
    """get_args gets the arguments for a function
    here it is tested on another function in this module"""
    import inspect  # used by get_args

    args = utils.get_args(utils.show_tensor)
    expected_args = {
        "tensor": inspect._empty,
        "channel": None,
        "normalize_from_range": [-1, 1],
        "invert": False,
        "cmap": None,
        "clip": [0, 1],
        "axis": None,
    }
    assert args == expected_args


def test_get_reqd_args():
    args = utils.get_reqd_args(utils.show_tensor)
    assert args == ["tensor"]


def test_show_tensor():
    """just assert that function runs without an error"""
    tensor = torch.empty((3, 224, 224))
    utils.show_tensor(tensor)


def test_show_tensor_wrong_channel():
    with pytest.raises(IndexError):
        tensor = torch.empty((1, 224, 224))
        utils.show_tensor(tensor, channel=2)  # channel 2 does not exist


def test_show_tensor_grid():
    tensors = [torch.empty((3, 224, 224)) for _ in range(12)]
    utils.show_tensor_grid(tensors, columns=3)


def test_show_tensor_grid_on_provided_axes():
    tensors = [torch.empty((3, 224, 224)) for _ in range(12)]
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(4, 3)
    utils.show_tensor_grid(tensors, columns=3, axes=axes)
