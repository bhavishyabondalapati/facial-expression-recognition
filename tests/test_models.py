"""Tests for the models, loss and augmentation (run on CPU, no dataset needed)."""
import torch
import torch.nn.functional as F

from src.models import build_model
from src.train import augment, soft_cross_entropy


def test_both_models_output_8_logits():
    x = torch.rand(2, 1, 48, 48)
    for name in ["cnn", "resnet18"]:
        model = build_model(name, pretrained=False).eval()
        assert model(x).shape == (2, 8)


def test_soft_ce_equals_standard_ce_for_one_hot_targets():
    logits = torch.randn(5, 8)
    labels = torch.tensor([0, 3, 7, 1, 1])
    one_hot = F.one_hot(labels, 8).float()
    assert torch.allclose(soft_cross_entropy(logits, one_hot), F.cross_entropy(logits, labels))


def test_soft_ce_is_lowest_when_prediction_matches_target():
    target = torch.tensor([[0.6, 0.4, 0, 0, 0, 0, 0, 0]])
    matching = torch.log(target.clamp_min(1e-6))      # logits that reproduce the target
    wrong = torch.zeros(1, 8)
    assert soft_cross_entropy(matching, target) < soft_cross_entropy(wrong, target)


def test_augment_keeps_shape_and_range():
    x = torch.rand(8, 1, 48, 48)
    y = augment(x, torch.Generator().manual_seed(0))
    assert y.shape == x.shape
    assert y.min() >= 0 and y.max() <= 1
