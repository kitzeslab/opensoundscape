# inspiration from zhmiao/BirdMultiLabel
import torch.nn as nn


class BaseArchitecture(nn.Module):

    """
    Base architecture for reference.
    """

    name = None

    def setup_net(self):
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
