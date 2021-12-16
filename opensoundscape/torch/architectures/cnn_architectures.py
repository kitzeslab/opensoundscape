"""Module to initialize PyTorch CNN architectures with custom output shape

This module allows the use of several built-in CNN architectures from PyTorch.
The architecture refers to the specific layers and layer input/output shapes
(including convolution sizes and strides, etc) - such as the ResNet18 or
Inception V3 architecture.

We provide wrappers which modify the output layer to the desired shape
(to match the number of classes). The way to change the output layer
shape depends on the architecture, which is why we need a wrapper for each one.
This code is based on
pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

To use these wrappers, for example, if your
model has 10 output classes, write

`my_arch=resnet18(10)`

Then you can initialize a model object from
`opensoundscape.torch.models.cnn` with your architecture:

`model=PytorchModel(my_arch,classes)`

or override an existing model's architecture:

`model.network = my_arch`

Note: the InceptionV3 architecture must be used differently than other
architectures - the easiest way is to simply use the InceptionV3 class in
opensoundscape.torch.models.cnn.
"""
from torchvision import models
from torch import nn
from opensoundscape.torch.architectures.utils import CompositeArchitecture

ARCH_DICT = dict()


def register_arch(func):
    # register the model in dictionary
    ARCH_DICT[func.__name__] = func
    # return the function
    return func


def set_parameter_requires_grad(model, freeze_feature_extractor):
    """if necessary, remove gradients of all model parameters

    if freeze_feature_extractor is True, we set requires_grad=False
    for all features in the feature extraction block. We would do this
    if we have a pre-trained CNN and only want to change the shape of the final
    layer, then train only that final classification layer without modifying
    the weights of the rest of the network.
    """
    if freeze_feature_extractor:
        for param in model.parameters():
            param.requires_grad = False


@register_arch
def resnet18(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for ResNet18 architecture

    input_size = 224

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


@register_arch
def resnet34(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for ResNet34 architecture

    input_size = 224

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """
    model_ft = models.resnet34(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


@register_arch
def resnet50(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for ResNet50 architecture

    input_size = 224

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """
    model_ft = models.resnet50(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


@register_arch
def resnet101(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for ResNet101 architecture

    input_size = 224

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """
    model_ft = models.resnet101(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


@register_arch
def resnet152(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for ResNet152 architecture

    input_size = 224

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """
    model_ft = models.resnet152(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


@register_arch
def alexnet(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for AlexNet architecture

    input size = 224

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """
    model_ft = models.alexnet(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model_ft


@register_arch
def vgg11_bn(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for vgg11 architecture

    input size = 224

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """

    model_ft = models.vgg11_bn(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model_ft


@register_arch
def squeezenet1_0(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for squeezenet architecture

    input size = 224

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    model_ft.classifier[1] = nn.Conv2d(
        512, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )
    model_ft.num_classes = num_classes
    return model_ft


@register_arch
def densenet121(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for densenet121 architecture

    input size = 224

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """
    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    return model_ft


@register_arch
def inception_v3(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Wrapper for Inception v3 architecture

    Input: 229x229

    WARNING: expects (299,299) sized images and has auxiliary output. See
    InceptionV3 class in `opensoundscape.torch.models.cnn` for use.

    Args:
        num_classes:
            number of output nodes for the final layer
        freeze_feature_extractor:
            if False (default), entire network will have gradients and can train
            if True, feature block is frozen and only final layer is trained
        use_pretrained:
            if True, uses pre-trained ImageNet features from
            Pytorch's model zoo.
    """
    model_ft = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    # Handle the auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def list_architectures():
    return list(ARCH_DICT.keys())
