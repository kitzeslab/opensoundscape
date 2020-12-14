# adapt from Miao's repository github.com/zhmiao/BirdMultiLabel
# this "model" is a custom resnet model architecture plus custom loss functions
# this .py file depends on (at least) a couple other things in the repository:
# - resnet backbone (utilities for resnet nn stuff?)
# - utility to register model in a list
# - BaseModule

# copying from https://github.com/zhmiao/BirdMultiLabel/blob/b31edf022e5c54a5d7ebe994460fec1579e90e96/src/models/distreg_resnet.py

import os
import copy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

from .utils import register_model, BaseModule
from .resnet_backbone import ResNetFeature, BasicBlock, Bottleneck, model_urls


@register_model("DistRegResNetClassifier")
class DistRegResNetClassifier(BaseModule):

    name = "DistRegResNetClassifier"

    def __init__(
        self,
        num_cls=10,
        weights_init="ImageNet",
        num_layers=18,
        init_feat_only=True,
        class_freq=None,
    ):
        super(DistRegResNetClassifier, self).__init__()
        self.num_cls = num_cls
        self.num_layers = num_layers
        self.feature = None
        self.classifier = None
        self.criterion_cls = None
        self.best_weights = None
        self.class_freq = class_freq

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
        self.criterion_cls = ResampleLoss(class_freq=self.class_freq)

    def load(self, init_path, feat_only=False):

        if "http" in init_path:
            init_weights = load_state_dict_from_url(init_path, progress=True)
        else:
            init_weights = torch.load(init_path)

        if feat_only:
            init_weights = OrderedDict(
                {k.replace("feature.", ""): init_weights[k] for k in init_weights}
            )
            self.feature.load_state_dict(init_weights, strict=False)
            load_keys = set(init_weights.keys())
            self_keys = set(self.feature.state_dict().keys())
        else:
            self.load_state_dict(init_weights, strict=False)
            load_keys = set(init_weights.keys())
            self_keys = set(self.state_dict().keys())

        missing_keys = self_keys - load_keys
        unused_keys = load_keys - self_keys
        print("missing keys: {}".format(sorted(list(missing_keys))))
        print("unused_keys: {}".format(sorted(list(unused_keys))))

    def save(self, out_path):
        torch.save(self.best_weights, out_path)

    def update_best(self):
        self.best_weights = copy.deepcopy(self.state_dict())


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == "mean":
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != "none":
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred, label, weight=None, reduction="mean", avg_factor=None):
    if pred.dim() != label.dim():
        if weight is not None:
            weight = weight.view(-1, 1).expand(weight.size(0), pred.size(-1))
        # label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction="none"
    )

    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss


class ResampleLoss(nn.Module):
    def __init__(self, class_freq, reduction="mean", loss_weight=1.0):
        super(ResampleLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

        self.cls_criterion = binary_cross_entropy

        # focal loss params
        self.focal = True
        self.gamma = 2
        self.balance_param = 2.0

        # mapping function params
        self.map_alpha = 10.0
        self.map_beta = 0.2
        self.map_gamma = 0.1

        self.class_freq = torch.from_numpy(class_freq).float().cuda()
        self.neg_class_freq = self.class_freq.sum() - self.class_freq
        self.num_classes = self.class_freq.shape[0]
        self.train_num = self.class_freq.sum()

        # regularization params
        # self.neg_scale = 2.0  # else 1.0
        # init_bias = 0.05  # else 0.0
        self.neg_scale = 0.2  # else 1.0
        init_bias = 0.05  # else 0.0
        self.init_bias = (
            -torch.log(self.train_num / self.class_freq - 1)
            * init_bias
            / self.neg_scale
        )
        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(
        self,
        cls_score,
        label,
        weight=None,
        avg_factor=None,
        reduction_override=None,
        **kwargs,
    ):

        assert reduction_override in (None, "none", "mean", "sum")

        reduction = reduction_override if reduction_override else self.reduction

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        loss = self.cls_criterion(cls_score, label.float(), weight, reduction=reduction)

        loss = self.loss_weight * loss

        return loss

    def reweight_functions(self, label):
        weight = self.rebalance_weight(label.float())
        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        logits += self.init_bias
        logits = logits * (1 - labels) * self.neg_scale + logits * labels
        weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = (
            torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma))
            + self.map_alpha
        )
        return weight
