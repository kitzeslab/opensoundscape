from opensoundscape.torch.architectures import cnn_architectures
import pytest


def test_freeze_feature_extractor():
    """should disable grad on featur extractor but not classifier"""
    arch = cnn_architectures.resnet18(2, freeze_feature_extractor=True)
    assert not arch.parameters().__next__().requires_grad
    assert arch.fc.parameters().__next__().requires_grad


def test_modify_resnet():
    """test modifying number of output nodes"""
    arch = cnn_architectures.resnet18(10)
    assert arch.fc.out_features == 10
