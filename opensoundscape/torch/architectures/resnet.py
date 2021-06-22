"""defines feature extractor and Architecture class for ResNet CNN

This implementation of the ResNet18 architecture allows for separate
access to the feature extraction and classification blocks. This can be
useful, for instance, to freeze the feature extractor and only train the
classifier layer; or to specify different learning rates for the two blocks.

This implementation is used in the Resnet18Binary and Resnet18Multiclass
classes of opensoundscape.torch.models.cnn.
"""
import os
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import conv1x1, BasicBlock, Bottleneck, model_urls

from opensoundscape.torch.architectures.utils import BaseArchitecture


class ResNetArchitecture(BaseArchitecture):
    """ResNet architecture with 18 or 50 layers

    This implementation enables separate access to feature and
    classification blocks.

    Args:
        num_cls: number of classes (int)
        weights_init:
            - "ImageNet": load the pre-trained weights for ImageNet dataset
            - path: load weights from a path on your computer or a url
            - None: initialize with random weights
        num_layers: 18 for Resnet18 or 50 for Resnet50
        init_classifier_weights:
            - if True, load the weights of the classification layer as well as
            feature extraction layers
            - if False (default), only load the weights of the feature extraction
            layers
    """

    name = "ResNetArchitecture"

    def __init__(
        self,
        num_cls,
        weights_init="ImageNet",
        num_layers=18,
        init_classifier_weights=False,
    ):

        super(ResNetArchitecture, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.feature = None
        self.classifier = None

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
            raise NameError("Initial weights do not exist: {}.".format(weights_init))

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

    def forward(self, batch_tensor):
        feats = self.feature(batch_tensor)  # feature extraction
        return self.classifier(feats)  # classification

    def load(self, init_path, init_classifier_weights=True, verbose=False):
        """load state dict (weights) of the feature+classifier
        optionally load only feature weights not classifier weights

        Args:
            init_path:
                - url containing "http": download weights from web
                - path: load weights from local path
            init_classifier_weights:
                - if True, load the weights of the classification layer as well as
                feature extraction layers
                - if False (default), only load the weights of the feature extraction
                layers
            verbose:
                if True, print missing/unused keys [default: False]
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


# Official ResNet Implementation
class ResNetFeature(nn.Module):
    def __init__(
        self,
        block,
        layers,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):

        super(ResNetFeature, self).__init__()

        ################
        # Model setups #
        ################
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        ########################
        # Model initialization #
        ########################
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
