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


class CompositeArchitecture(nn.Module):
    """Architecture with separate feature and classsifier blocks"""

    def forward(self, batch_tensor):
        feats = self.feature(batch_tensor)  # feature extraction
        return self.classifier(feats)  # classification
