import torch

from opensoundscape.ml.loss import BCELossWeakNegatives


def test_bce_weak_negatives_matches_bce_without_nans():
    """Verify BCELossWeakNegatives is equivalent to standard BCEWithLogitsLoss when no NaN labels present.

    When all targets are valid (no NaN), BCELossWeakNegatives should produce identical loss
    to the standard PyTorch BCEWithLogitsLoss, since no weak-negative weighting is applied.
    """
    x = torch.tensor([[0.1, -0.4], [1.0, -2.0]], dtype=torch.float32)
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

    loss_fn = BCELossWeakNegatives(weak_negative_weight=0.25)
    weak_loss = loss_fn(x, target)

    baseline = torch.nn.BCEWithLogitsLoss()(x, target)
    assert torch.allclose(weak_loss, baseline, atol=1e-8)


def test_bce_weak_negatives_downweights_nan_targets():
    """Verify BCELossWeakNegatives applies reduced weight to NaN labels.

    When target contains NaN (representing unlabeled/ambiguous samples), the loss for that
    label should be downweighted relative to labeled samples. This test verifies the math:
    NaN is converted to 0 in the target, weighted by weak_negative_weight, then the loss
    is normalized by the sum of all weights.
    """
    x = torch.tensor([[0.0, 0.5]], dtype=torch.float32)
    target = torch.tensor([[1.0, float("nan")]], dtype=torch.float32)
    weak_w = 0.1

    loss_fn = BCELossWeakNegatives(weak_negative_weight=weak_w)
    loss = loss_fn(x, target)

    # Manual expectation: nan is treated as 0 target, then weighted and normalized.
    per_item = torch.nn.functional.binary_cross_entropy_with_logits(
        x,
        torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        reduction="none",
    )
    expected = (per_item[0, 0] * 1.0 + per_item[0, 1] * weak_w) / (1.0 + weak_w)

    assert torch.allclose(loss, expected, atol=1e-8)


def test_bce_weak_negatives_accepts_integer_targets():
    """Verify BCELossWeakNegatives accepts integer targets and converts them to float.

    The loss function should handle both integer (long) and float target types by
    internally converting them to float before processing. This is important for
    compatibility with various label encoding formats.
    """
    x = torch.tensor([[0.2, -0.2], [0.3, -0.3]], dtype=torch.float32)
    target = torch.tensor([[1, 0], [0, 1]], dtype=torch.int64)

    loss = BCELossWeakNegatives()(x, target)
    assert torch.isfinite(loss)


def test_bce_weak_negatives_accepts_bool_targets():
    x = torch.tensor([[0.2, -0.2], [0.3, -0.3]], dtype=torch.float32)
    target = torch.tensor([[True, False], [False, True]], dtype=torch.bool)

    loss = BCELossWeakNegatives()(x, target)
    assert torch.isfinite(loss)
