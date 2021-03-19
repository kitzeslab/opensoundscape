import os
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from opensoundscape.torch.architectures.utils import BaseArchitecture
from opensoundscape.torch.architectures.resnet_backbone import (
    ResNetFeature,
    BasicBlock,
    Bottleneck,
    model_urls,
)

# @register_model('PlainResNetClassifier')
class PlainResNetClassifier(BaseArchitecture):

    name = "PlainResNetClassifier"

    def __init__(
        self, num_cls, weights_init="ImageNet", num_layers=18, init_feat_only=True
    ):

        super(PlainResNetClassifier, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.feature = None
        self.classifier = None
        self.criterion_cls = None
        self.best_weights = None

        # Model setup and weights initialization
        self.setup_net()
        if weights_init == "ImageNet":
            self.load(
                model_urls["resnet{}".format(num_layers)], feat_only=init_feat_only
            )
        elif os.path.exists(weights_init):
            self.load(weights_init, feat_only=init_feat_only)
        elif weights_init != "ImageNet" and not os.path.exists(weights_init):
            raise NameError("Initial weights not exists {}.".format(weights_init))

        # Criteria setup
        self.setup_critera()

    def setup_net(self):

        kwargs = {}

        if self.num_layers == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif self.num_layers == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        else:
            raise Exception("ResNet Type not supported.")

        self.feature = ResNetFeature(block, layers, **kwargs)
        self.classifier = nn.Linear(512 * block.expansion, self.num_cls)

    def setup_critera(self):
        self.criterion_cls = BCEWithLogitsLoss_hot()

    # def load_url(self,weights_url,feat_only=False):
    #     init_weights = load_state_dict_from_url(weights_url, progress=True)
    #     self.load_weights(init_weights, feat_only)

    # def load_weights(self, init_weights, feat_only=False):
    #     #
    #     # #we will provide the init_weights dictionary instead of loading it from disk
    #     # if 'http' in init_path:
    #     #     init_weights = load_state_dict_from_url(init_path, progress=True)
    #     # else:
    #     #     init_weights = torch.load(init_path)
    #
    #     if feat_only:
    #         init_weights = OrderedDict({k.replace('feature.', ''): init_weights[k]
    #                                     for k in init_weights})
    #         self.feature.load_state_dict(init_weights, strict=False)
    #         load_keys = set(init_weights.keys())
    #         self_keys = set(self.feature.state_dict().keys())
    #     else:
    #         self.load_state_dict(init_weights, strict=False)
    #         load_keys = set(init_weights.keys())
    #         self_keys = set(self.state_dict().keys())
    #
    #     missing_keys = self_keys - load_keys
    #     unused_keys = load_keys - self_keys
    #     print("missing keys: {}".format(sorted(list(missing_keys))))
    #     print("unused_keys: {}".format(sorted(list(unused_keys))))
    #
    # def save(self, out_path):
    #     torch.save(self.best_weights, out_path)
    #
    # def update_best(self):
    #     self.best_weights = copy.deepcopy(self.state_dict())


class BCEWithLogitsLoss_hot(nn.BCEWithLogitsLoss):
    """use nn.BCEWithLogitsLoss for one-hot labels
    by simply converting y from long to float"""

    def __init__(self):
        super(BCEWithLogitsLoss_hot, self).__init__()

    def forward(self, input, target):
        target = target.float()
        return super(BCEWithLogitsLoss_hot, self).forward(input, target)
