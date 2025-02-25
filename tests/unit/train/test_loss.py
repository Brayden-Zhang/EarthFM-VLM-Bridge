"""Test losses."""

import torch

from helios.nn.flexihelios import TokensAndMasks
from helios.train.loss import CrossEntropyLoss, L1Loss, L2Loss, PatchDiscriminationLoss


def test_patch_disc_loss() -> None:
    """Just test that it runs as expected."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_mask=torch.ones((b, t_h, t_w, t)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = PatchDiscriminationLoss()
    loss_value = loss.compute(preds, targets)
    # not very good! since they are all the same
    # predictions and values
    assert loss_value > 0.5


def test_patch_disc_loss_averaged_over_batch_size() -> None:
    """Test it doesn't scale with batch size."""
    b, t_h, t_w, t, d = 3, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_mask=torch.ones((b, t_h, t_w, t)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = PatchDiscriminationLoss()
    loss_value = loss.compute(preds, targets)

    # now, use a larger batch size
    b, t_h, t_w, t, d = 8, 4, 4, 2, 2

    preds = TokensAndMasks(
        sentinel2=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_mask=torch.ones((b, t_h, t_w, t)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2=torch.ones((b, t_h, t_w, t, d)),
        sentinel2_mask=torch.zeros((b, t_h, t_w, t)),
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss_value_8 = loss.compute(preds, targets)
    # not very good! since they are all the same
    # predictions and values
    assert torch.isclose(loss_value, loss_value_8)


def test_l1_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2=torch.ones((b, t, t_h, t_w, d)),
        sentinel2_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2=torch.zeros((b, t, t_h, t_w, d)),
        sentinel2_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = L1Loss()
    loss_value = loss.compute(preds, targets)
    # MAE should be 1 since preds are 1, targets are 0
    assert loss_value == 1


def test_l2_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2=2 * torch.ones((b, t, t_h, t_w, d)),
        sentinel2_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=2 * torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2=torch.zeros((b, t, t_h, t_w, d)),
        sentinel2_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = L2Loss()
    loss_value = loss.compute(preds, targets)
    # MSE should be 4 since preds are 2, targets are 0
    assert loss_value == 4

def test_l2_loss_is_same_with_gradient_accumulation(set_random_seeds: None) -> None:
    """Test that the loss is the same with gradient accumulation."""
    b, t, t_h, t_w, d = 4, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2=2 * torch.ones((b, t, t_h, t_w, d)),
        sentinel2_mask=torch.randint(0, 3, (b, t, t_h, t_w)),
        latlon=2 * torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2=torch.zeros((b, t, t_h, t_w, d)),
        sentinel2_mask=torch.randint(0, 3, (b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, d)),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = L2Loss()
    loss_value = loss.compute(preds, targets)
    # Now slice the batch into 2 smaller batches
    preds_1 = TokensAndMasks(
        sentinel2=preds.sentinel2[:2],
        sentinel2_mask=preds.sentinel2_mask[:2],
        latlon=preds.latlon[:2],
        latlon_mask=preds.latlon_mask[:2],
    )
    targets_1 = TokensAndMasks(
        sentinel2=targets.sentinel2[:2],
        sentinel2_mask=targets.sentinel2_mask[:2],
        latlon=targets.latlon[:2],
        latlon_mask=targets.latlon_mask[:2],
    )
    preds_2 = TokensAndMasks(
        sentinel2=preds.sentinel2[2:],
        sentinel2_mask=preds.sentinel2_mask[2:],
        latlon=preds.latlon[2:],
        latlon_mask=preds.latlon_mask[2:],
    )
    targets_2 = TokensAndMasks(
        sentinel2=targets.sentinel2[2:],
        sentinel2_mask=targets.sentinel2_mask[2:],
        latlon=targets.latlon[2:],
        latlon_mask=targets.latlon_mask[2:],
    )
    loss_value_1 = loss.compute(preds_1, targets_1)
    loss_value_2 = loss.compute(preds_2, targets_2)
    loss_value = (loss_value_1 + loss_value_2) / 2
    assert torch.isclose(loss_value, loss_value_1)


def test_cross_entropy_loss() -> None:
    """Just test that it runs as expected."""
    b, t, t_h, t_w, d = 3, 2, 4, 4, 2

    preds = TokensAndMasks(
        sentinel2=2 * torch.ones((b, t, t_h, t_w, d)),
        sentinel2_mask=torch.ones((b, t, t_h, t_w)) * 2,
        latlon=2 * torch.ones((b, 1, d)),
        latlon_mask=torch.ones((b, 1)) * 2,
    )
    targets = TokensAndMasks(
        sentinel2=torch.zeros((b, t, t_h, t_w, 1), dtype=torch.long),
        sentinel2_mask=torch.zeros((b, t, t_h, t_w)),
        latlon=torch.zeros((b, 1, 1), dtype=torch.long),
        latlon_mask=torch.zeros((b, 1)),
    )
    loss = CrossEntropyLoss()
    loss_value = loss.compute(preds, targets)
    # loss for BCE, prediction of .5 for both classes
    assert torch.isclose(loss_value, -torch.log(torch.tensor(0.5)), 0.0001)
