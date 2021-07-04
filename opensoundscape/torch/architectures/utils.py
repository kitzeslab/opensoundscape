# copying from https://github.com/zhmiao/BirdMultiLabel/blob/b31edf022e5c54a5d7ebe994460fec1579e90e96/src/models/utils.py

import torch
import torch.nn as nn


class BaseArchitecture(nn.Module):

    """
    Base architecture for reference.
    """

    name = None

    def __init__(self):
        super(BaseArchitecture, self).__init__()

    def setup_net(self):
        pass

    def load(self, init_path):
        pass

    def save(self, out_path):
        pass

    def update_best(self):
        pass


class Identity(nn.Module):
    """A layer that returns the input without modification

    Used for 'removing' the classification layer of a network
    (by replacing it with the identity transform X->X)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def get_output_shape(model, image_dim):
    return model(torch.rand(*(image_dim))).data.shape
