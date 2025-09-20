# test generating cam
# test plotting and saving

# unit tests for CAM class
import pytest
import torch
import numpy as np
from opensoundscape.ml.cam import CAM
import pandas as pd


@pytest.fixture()
def cam():
    base = torch.rand(1, 224, 224)
    activation_maps = pd.Series(
        {i: np.random.uniform(0, 1, [224, 224]) for i in range(2)}
    )
    gbp_maps = pd.Series({i: np.random.uniform(0, 1, [224, 224]) for i in range(2)})
    return CAM(
        base_image=base,
        activation_maps=activation_maps,
        gbp_maps=gbp_maps,
    )


def test_cam_init(cam):
    assert isinstance(cam, CAM)
    assert isinstance(cam.base_image, torch.Tensor)
    assert isinstance(cam.activation_maps, pd.Series)
    assert isinstance(cam.gbp_maps, pd.Series)


def test_cam_plot(cam):
    cam.plot()
    cam.plot(class_subset=[0])
    cam.plot(class_subset=(0, 1))
    cam.plot(class_subset=(0,), mode="backprop")
    cam.plot(class_subset=(0,), mode="backprop_and_activation")


def test_cam_plot_None(cam):
    cam.plot(mode=None)
