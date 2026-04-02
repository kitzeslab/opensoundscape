import torch
import torch.nn as nn
import pytest

from opensoundscape.ml.base_model import BaseModule
from opensoundscape.preprocess.preprocessors import BasePreprocessor
from opensoundscape.ml.dataloaders import SafeAudioDataloader


def test_base_module_init_defaults():
    model = BaseModule()

    assert model.name == "BaseModule"
    assert isinstance(model.preprocessor, BasePreprocessor)
    assert model.train_dataloader_cls is SafeAudioDataloader
    assert model.inference_dataloader_cls is SafeAudioDataloader
    assert "accuracy" in model.torch_metrics
    assert model.loss_fn is not None
    assert model.optimizer_params["class"] is torch.optim.SGD


def test_predict_dataloader_wraps_single_path_and_sets_fixed_args():
    model = BaseModule()
    model.preprocessor = object()
    model.device = torch.device("cpu")
    captured = {}

    class FakeLoader:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    model.inference_dataloader_cls = FakeLoader
    model.predict_dataloader(
        "tests/audio/silence_10s.mp3",
        raise_errors=True,
        batch_size=3,
    )

    assert captured["samples"] == ["tests/audio/silence_10s.mp3"]
    assert captured["shuffle"] is False
    assert captured["pin_memory"] is False
    assert captured["invalid_sample_behavior"] == "raise"
    assert captured["batch_size"] == 3


def test_configure_optimizers_returns_optimizer_and_scheduler():
    model = BaseModule()
    model.network = nn.Linear(4, 2)
    model.lr_scheduler_step = -1

    out = model.configure_optimizers()

    assert "optimizer" in out
    assert "scheduler" in out
    assert isinstance(out["optimizer"], torch.optim.Optimizer)


def test_configure_optimizers_classifier_lr_creates_second_param_group():
    model = BaseModule()
    model.network = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    model.classifier = model.network[1]
    model.optimizer_params["classifier_lr"] = 0.123
    model.lr_scheduler_step = -1

    out = model.configure_optimizers()
    optimizer = out["optimizer"]
    lrs = {group["lr"] for group in optimizer.param_groups}

    assert len(optimizer.param_groups) == 2
    assert 0.123 in lrs
    assert model.optimizer_params["kwargs"]["lr"] in lrs


def test_configure_optimizers_classifier_lr_without_classifier_raises_value_error():
    model = BaseModule()
    model.network = nn.Linear(4, 2)
    model.optimizer_params["classifier_lr"] = 0.123
    model.lr_scheduler_step = -1

    with pytest.raises(ValueError, match="self.classifier"):
        model.configure_optimizers()
