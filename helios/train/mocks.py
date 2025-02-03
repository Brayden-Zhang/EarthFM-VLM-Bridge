"""Mock classes for the training module.

This module implements extensible abstractions for masking strategies and loss functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Tuple, TypeVar

from olmo_core.config import Config

# Define a generic type variable for input data types
TData = TypeVar("TData")


# Masking Needs
# 1. Apply many different strategies
# 2. Randomly apply some combination of these strategies
# 3. Configure and build the object that does the masking that can then be passed to the train module together with the corresponding model
# 4. Loss class should be able to employ a variety of different loss functions and combinations thereof
#  a bunch of sperate implementations and then a meta class that can combine them

# Masker class
# list of masking strategies
# Setting on how to apply them
# Each strategy handles masking for all the modalities


class MaskingStrategy(ABC, Generic[TData]):
    """Abstract base class for masking strategies."""

    @abstractmethod
    def mask(self, data: TData, **kwargs) -> Tuple[TData, Any]:
        """Apply masking to the input data.

        Args:
            data: Input data of type TData
            **kwargs: Additional arguments for maskings

        Returns:
            Tuple of (masked_data, mask)
        """
        pass


class CompositeMaskingStrategy(MaskingStrategy[TData]):
    """Combines multiple masking strategies with configurable application logic."""

    def __init__(
        self,
        strategies: list[MaskingStrategy[TData]],
        probabilities: list[float] | None = None,
    ):
        """
        Args:
            strategies: List of masking strategies to combine
            probabilities: Optional list of probabilities for each strategy.
                         If None, strategies are applied sequentially.
        """
        self.strategies = strategies
        self.probabilities = probabilities
        if probabilities and len(probabilities) != len(strategies):
            raise ValueError("Number of probabilities must match number of strategies")

    def mask(self, data: TData, **kwargs) -> Tuple[TData, list[Any]]:
        """Apply multiple masking strategies to the input data.

        Returns:
            Tuple of (masked_data, list of masks)
        """
        current_data = data
        masks = []

        if self.probabilities:
            import random

            for strategy, prob in zip(self.strategies, self.probabilities):
                if random.random() < prob:
                    current_data, mask = strategy.mask(current_data, **kwargs)
                    masks.append(mask)
        else:
            for strategy in self.strategies:
                current_data, mask = strategy.mask(current_data, **kwargs)
                masks.append(mask)

        return current_data, masks


class Loss(ABC):
    """Abstract base class for loss functions."""

    @abstractmethod
    def compute(self, predictions: Any, targets: Any, **kwargs) -> float:
        """Compute the loss between predictions and targets."""
        pass


class CompositeLoss:
    """Combines multiple loss functions with optional weights."""

    def __init__(self, losses: list[Loss], weights: list[float] | None = None):
        """
        Args:
            losses: List of loss functions to combine
            weights: Optional weights for each loss function.
                    If None, losses are weighted equally.
        """
        self.losses = losses
        self.weights = weights or [1.0] * len(losses)

        if len(self.weights) != len(losses):
            raise ValueError("Number of weights must match number of losses")

    def compute(self, predictions: Any, targets: Any, **kwargs) -> float:
        """Compute weighted combination of multiple losses.

        Returns:
            Combined loss value
        """
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn.compute(predictions, targets, **kwargs)
        return total_loss


@dataclass
class MaskingConfig(Config):
    """Configuration for masking strategies."""

    strategies: list[dict[str, Any]]  # List of strategy configs
    probabilities: list[float] | None = None

    def validate(self):
        """Validate the masking configuration."""
        if self.probabilities and len(self.probabilities) != len(self.strategies):
            raise ValueError("Number of probabilities must match number of strategies")

    def build(self) -> CompositeMaskingStrategy:
        """Build a CompositeMaskingStrategy from the config."""
        built_strategies = []
        for strategy_config in self.strategies:
            strategy_type = strategy_config.pop("type")
            # This assumes you have a registry of strategies or will implement specific builders
            strategy = MASKING_STRATEGY_REGISTRY[strategy_type](**strategy_config)
            built_strategies.append(strategy)

        return CompositeMaskingStrategy(
            strategies=built_strategies, probabilities=self.probabilities
        )


@dataclass
class LossConfig(Config):
    """Configuration for loss functions."""

    losses: list[dict[str, Any]]  # List of loss configs
    weights: list[float] | None = None

    def validate(self):
        """Validate the loss configuration."""
        if self.weights and len(self.weights) != len(self.losses):
            raise ValueError("Number of weights must match number of losses")

    def build(self) -> CompositeLoss:
        """Build a CompositeLoss from the config."""
        built_losses = []
        for loss_config in self.losses:
            loss_type = loss_config.pop("type")
            # This assumes you have a registry of losses or will implement specific builders
            loss = LOSS_REGISTRY[loss_type](**loss_config)
            built_losses.append(loss)

        return CompositeLoss(losses=built_losses, weights=self.weights)


# Registry dictionaries to be populated with available strategies and losses
MASKING_STRATEGY_REGISTRY: dict[str, type[MaskingStrategy]] = {}
LOSS_REGISTRY: dict[str, type[Loss]] = {}

# pass each of these configs to train module to be built