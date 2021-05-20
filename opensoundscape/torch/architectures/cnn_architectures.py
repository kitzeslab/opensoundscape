"""Module to initialize pytorch CNN architectures with custom output shape

We provide wrappers which modify the output layer to the desired shape
(to match the number of classes). The way to change the output layer
shape depends on the architecture, so this should make it easier to swap out
architectures. Based on
pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

"""
from torchvision import models
from torch import nn


def set_parameter_requires_grad(model, freeze_feature_extractor):
    """if freeze_feature_extractor is True, we set requires_grad=False
    for all features in the feature extraction block. We would do this
    if we have a pre-trained CNN and only want to change the shape of the final
    layer, then train only that final classification layer."""
    if freeze_feature_extractor:
        for param in model.parameters():
            param.requires_grad = False


def resnet18(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """input_size = 224

    use feature_block_requires_grad=False if we want to freeze the feature block
    and only train the classification block of the network
    """
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    return model_ft


def alexnet(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """input size = 224"""
    model_ft = models.alexnet(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model_ft


def vgg11_bn(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """input size = 224"""
    model_ft = models.vgg11_bn(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    return model_ft


def squeezenet1_0(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """input size = 224"""
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    model_ft.classifier[1] = nn.Conv2d(
        512, num_classes, kernel_size=(1, 1), stride=(1, 1)
    )
    model_ft.num_classes = num_classes
    return model_ft


def densenet121(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """input size = 224"""
    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, freeze_feature_extractor)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    return model_ft


def inception_v3(num_classes, freeze_feature_extractor=False, use_pretrained=True):
    """Inception v3
    Warning: expects (299,299) sized images and has auxiliary output. Will
    require specialized training and prediction code to use.
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


# TODO: add other architectures
# import torchvision.models as models
# resnet18 = models.resnet18()
# alexnet = models.alexnet()
# vgg16 = models.vgg16()
# squeezenet = models.squeezenet1_0()
# densenet = models.densenet161()
# inception = models.inception_v3()
# googlenet = models.googlenet()
# shufflenet = models.shufflenet_v2_x1_0()
# mobilenet_v2 = models.mobilenet_v2()
# mobilenet_v3_large = models.mobilenet_v3_large()
# mobilenet_v3_small = models.mobilenet_v3_small()
# resnext50_32x4d = models.resnext50_32x4d()
# wide_resnet50_2 = models.wide_resnet50_2()
# mnasnet = models.mnasnet1_0()
