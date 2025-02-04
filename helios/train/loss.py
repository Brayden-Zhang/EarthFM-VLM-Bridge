"""Loss functions for training."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from class_registry import ClassRegistry
from olmo_core.config import Config

from helios.train.model import TokensAndMasks


# V-1 loss function
def patch_disc_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: float = 0.2,
    pred2unit: bool = True,
) -> torch.Tensor:
    """Patch discriminator loss.

    This function computes the discriminative loss between predicted and target patches.

    Args:
        pred: Predicted patches.
        target: Target patches.
        tau: Temperature parameter for the loss.
        pred2unit: Whether to normalize the predicted patches to unit length.

    Returns:
        Loss tensor.
    """
    # Input shape: (B, N, C)
    # Target shape: (B, N, C)
    # B is batch size
    # N is number of patches
    # C is embedding dimension
    #
    B, N, C = pred.shape

    if pred2unit:
        pred_mu = pred.mean(1, keepdims=True)
        pred_std = pred.std(1, keepdims=True)
        pred = (pred - pred_mu) / (pred_std + 1e-4)

    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)

    # n is batch dimension p is patch index from pred q is patch index from target, d is embedding dimension
    scores = torch.einsum("npd,nqd->npq", pred, target) / tau

    labels = torch.arange(N, dtype=torch.long, device=pred.device)[None].repeat(B, 1)
    # Target is the index of the patch in the target so we are aiming to make the scores from the same patch similar and different patches dissimilar
    loss = F.cross_entropy(
        scores.flatten(0, 1),
        labels.flatten(0, 1),
    ) * (tau * 2)

    return loss


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def compute(self, predictions: Any, targets: Any, **kwargs: Any) -> float:
        """Compute the loss between predictions and targets."""
        pass


LOSS_REGISTRY = ClassRegistry[Loss]()


@LOSS_REGISTRY.register("patch_discrimination")
class PatchDiscriminationLoss(Loss):
    """Loss function for patch discrimination task."""

    def __init__(self, tau: float = 0.07, pred2unit: bool = True):
        """Initialize patch discrimination loss."""
        self.tau = tau
        self.pred2unit = pred2unit

    def compute(
        self, predictions: TokensAndMasks, targets: TokensAndMasks, **kwargs: Any
    ) -> float:
        """Compute patch discrimination loss between predictions and targets.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets.
            **kwargs: Additional keyword arguments.

        Returns:
            The computed loss value.

        Raises:
            NotImplementedError: This method needs to be implemented.
        """
        raise NotImplementedError


@dataclass
class LossConfig(Config):
    """Configuration for loss functions.

    Args:
        loss_config: Loss config in the format of
        e.g.
        {
            "type": "patch_discrimination",
            # rest of init kwargs
    """

    loss_config: dict[str, Any]  # List of loss configs

    def build(self) -> type[Loss]:
        """Build a Loss from the config."""
        return LOSS_REGISTRY[self.loss_config["type"]](**self.loss_config)
