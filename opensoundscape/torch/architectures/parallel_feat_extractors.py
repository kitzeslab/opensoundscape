"""defines Architecture class that allows multiple feature extractors

The features are concatenated and passed to a classification layer
"""
import os
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from opensoundscape.torch.architectures.utils import BaseArchitecture, get_output_shape


class MultiFeatureArchitecture(BaseArchitecture):
    """Parent class for architectures with parallel feature extractors

    This implementation enables you to pass an arbitrary set of feature
    extraction blocks. The features are each computed separately for a sample,
    then concatenated and passed to a fully-connected classification layer.

    Args:
        num_cls: number of classes (int)
        feature_extractors: list of feature extraction architectures
            - each should subclass nn.Module and have a .forward() method

    """

    name = "MultiFeatureArchitecture"

    def __init__(self, num_cls, feature_extractors, sample_shape=[3, 224, 224]):  # list

        super(MultiFeatureArchitecture, self).__init__()
        self.num_cls = num_cls
        self.feature = None
        self.classifier = None

        # Model setup and weights initialization
        # first check the output size of each feature extractor
        shape = [1] + sample_shape
        out_shapes = [get_output_shape(fe, shape)[1] for fe in feature_extractors]
        self.num_feat = sum(out_shapes)
        self.feature_extractors = feature_extractors
        self.classifier = nn.Linear(self.num_feat, self.num_cls)

    def forward(self, batch_tensor):
        # calculate features from each feature extractor
        feats = [fe(batch_tensor) for fe in self.feature_extractors]
        # concatenate into a single feature vector
        feats = torch.cat(feats, 1)

        # run classifier block
        return self.classifier(feats)
