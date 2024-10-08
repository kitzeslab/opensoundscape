from opensoundscape.ml import cnn_architectures
import pytest
import torch

# test_cnn.py tests that all registered architectures are able to
# predict on a sample (with modified input channels and output size)


def test_freeze_feature_extractor():
    """should disable grad on featur extractor but not classifier"""
    arch = cnn_architectures.resnet18(2, freeze_feature_extractor=True)
    assert not arch.parameters().__next__().requires_grad
    assert arch.fc.parameters().__next__().requires_grad


def test_modify_out_shape():
    """test modifying number of output nodes (classes)"""
    arch = cnn_architectures.resnet18(10)
    assert arch.fc.out_features == 10


def test_freeze_params():
    """tests that model parameters are frozen"""
    arch = cnn_architectures.resnet18(100)
    cnn_architectures.freeze_params(arch)
    for param in arch.parameters():
        assert param.requires_grad == False


def test_resnet18():
    arch = cnn_architectures.resnet18(num_classes=2, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_resnet34():
    arch = cnn_architectures.resnet34(10, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_resnet50():
    arch = cnn_architectures.resnet50(2000, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_resnet101():
    arch = cnn_architectures.resnet101(4, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_resnet152():
    arch = cnn_architectures.resnet152(3, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_alexnet():
    arch = cnn_architectures.alexnet(2, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_vgg11_bn():
    arch = cnn_architectures.vgg11_bn(2, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_squeezenet1_0():
    arch = cnn_architectures.squeezenet1_0(10, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_densenet121():
    arch = cnn_architectures.densenet121(111, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_inception_v3():
    arch = cnn_architectures.inception_v3(1, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_use_no_weights():
    arch = cnn_architectures.resnet50(2000, weights=None)
    assert isinstance(arch, torch.nn.Module)


def test_load_specific_weights():
    arch = cnn_architectures.resnet101(4, weights="IMAGENET1K_V2")
    assert isinstance(arch, torch.nn.Module)


def test_efficientnet_noweights():
    arch = cnn_architectures.efficientnet_b0(2, weights=None)
    assert isinstance(arch, torch.nn.Module)


def test_efficientnet_b0():
    arch = cnn_architectures.efficientnet_b0(2, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_efficientnet_b4():
    arch = cnn_architectures.efficientnet_b4(2, weights="DEFAULT")
    assert isinstance(arch, torch.nn.Module)


def test_noninteger_output_nodes():
    with pytest.raises(TypeError):
        arch = cnn_architectures.resnet101(4.5)
