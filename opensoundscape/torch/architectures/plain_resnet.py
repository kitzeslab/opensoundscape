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


class PlainResNetClassifier(BaseArchitecture):

    name = "PlainResNetClassifier"

    def __init__(
        self,
        num_cls,
        weights_init="ImageNet",
        num_layers=18,
        init_classifier_weights=False,
    ):

        super(PlainResNetClassifier, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.feature = None
        self.classifier = None
        self.criterion_cls = None
        self.best_weights = None

        # Model setup and weights initialization
        # if init_classifier_weights=False: copy feature weights but not classifier weights
        # ie if we want to re-use trained feature extractor w/ different classifier
        self.setup_net()
        if weights_init == "ImageNet":
            self.load(
                model_urls["resnet{}".format(num_layers)],
                init_classifier_weights=init_classifier_weights,
            )
        elif os.path.exists(weights_init):  # load weights from disk
            self.load(weights_init, init_classifier_weights=init_classifier_weights)
        else:
            raise NameError("Initial weights not exists {}.".format(weights_init))

        # Criteria (loss function) setup
        self.setup_loss()

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

    def setup_loss(self):
        self.criterion_cls = BCEWithLogitsLoss_hot()

    def load(self, init_path, init_classifier_weights=True, verbose=False):
        """load state dict (weights) of the feature+classifier
        optionally load only feature weights not classifier weights

        if verbose==True: print missing/unused keys
        """

        if "http" in init_path:
            init_weights = load_state_dict_from_url(init_path, progress=True)
        else:
            init_weights = torch.load(init_path)

        if init_classifier_weights:  # load all weights
            self.load_state_dict(init_weights, strict=False)
            load_keys = set(init_weights.keys())
            self_keys = set(self.state_dict().keys())
        else:  # only load feature weights not classifier weights
            init_weights = OrderedDict(
                {k.replace("feature.", ""): init_weights[k] for k in init_weights}
            )  # remove prefix
            self.feature.load_state_dict(init_weights, strict=False)
            load_keys = set(init_weights.keys())
            self_keys = set(self.feature.state_dict().keys())

        if verbose:
            # check if some weight_dict keys were missing or unused
            missing_keys = self_keys - load_keys
            unused_keys = load_keys - self_keys
            print("missing keys: {}".format(sorted(list(missing_keys))))
            print("unused_keys: {}".format(sorted(list(unused_keys))))

    def save(self, out_path):
        # we don't use this anymore?
        torch.save(self.best_weights, out_path)

    def update_best(self):
        self.best_weights = copy.deepcopy(self.state_dict())


class BCEWithLogitsLoss_hot(nn.BCEWithLogitsLoss):
    """use nn.BCEWithLogitsLoss for one-hot labels
    by simply converting y from long to float"""

    def __init__(self):
        super(BCEWithLogitsLoss_hot, self).__init__()

    def forward(self, input, target):
        target = target.float()
        return super(BCEWithLogitsLoss_hot, self).forward(input, target)
