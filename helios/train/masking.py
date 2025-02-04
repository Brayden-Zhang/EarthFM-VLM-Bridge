"""Masking module."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NamedTuple

from class_registry import ClassRegistry
from olmo_core.config import Config

from helios.data.dataset import HeliosSample
from helios.helios.types import ArrayTensor


# SHould be return type of masking strategy
class MaskedHeliosSample(NamedTuple):
    """A masked sample of the data from the Helios dataset.

    This is a namedtuple that contains the data for a single sample from the Helios dataset.
    For each modality. we have an ArrayTensor named by modality, positions in lat lon named modality_latlon, and
    time named modality_timestamps, as well as a mask for each of these modalities.

    Args:
        s2: ArrayTensor  # [B, len(S2_bands), T H, W]
        s2_mask: ArrayTensor
        s2_latlon: ArrayTensor  # [B, 2]
        s2_latlon_mask: ArrayTensor
        s2_timestamps: ArrayTensor  # [B, D=3, T], where D=[day, month, year]
    """

    s2: ArrayTensor  # [B, len(S2_bands), T H, W]
    s2_mask: ArrayTensor
    s2_latlon: ArrayTensor  # [B, 2]
    s2_latlon_mask: ArrayTensor
    s2_timestamps: ArrayTensor  # [B, D=3, T], where D=[day, month, year]

    def as_dict(self) -> dict[str, Any]:
        """Convert the namedtuple to a dictionary.

        Returns:
            Dictionary representation of the namedtuple.
        """
        return_dict = {}
        for field in self._fields:
            val = getattr(self, field)
            return_dict[field] = val
        return return_dict

    @property
    def height(self) -> int:
        """Get the height of the data."""
        return self.s2.shape[2]

    @property
    def width(self) -> int:
        """Get the width of the data."""
        return self.s2.shape[3]

    @staticmethod
    def get_masked_modality_name(modality: str) -> str:
        """Get the masked modality name."""
        return f"{modality}_mask"


class MaskingStrategy(ABC):
    """Abstract base class for masking strategies."""

    @abstractmethod
    def apply_mask(self, batch: HeliosSample, **kwargs: Any) -> MaskedHeliosSample:
        """Apply masking to the input data.

        Args:
            batch: Input data of type HeliosSample
            **kwargs: Additional arguments for maskings

        Returns:
            Tuple of (masked_data, mask)
        """
        pass


MASKING_STRATEGY_REGISTRY = ClassRegistry[MaskingStrategy]()


# EXAMPLE
@MASKING_STRATEGY_REGISTRY.register("random")
class RandomMaskingStrategy(MaskingStrategy):
    """Randomly masks the input data."""

    def apply_mask(self, batch: HeliosSample, **kwargs: Any) -> MaskedHeliosSample:
        """Apply random masking to the input data.

        Args:
            batch: Input data of type HeliosSample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedHeliosSample containing the masked data and mask
        """
        raise NotImplementedError


class CompositeMaskingStrategy(MaskingStrategy):
    """Combines multiple masking strategies with configurable application logic."""

    def __init__(
        self,
        strategies: list[MaskingStrategy],
        probabilities: list[float] | None = None,
    ):
        """Initialize a composite masking strategy.

        Args:
            strategies: List of masking strategies to combine
            probabilities: Optional list of probabilities for each strategy.
                         If None, strategies are applied sequentially.
        """
        self.strategies = strategies
        self.probabilities = probabilities
        if len(self.strategies) > 1:
            if self.probabilities is None:
                # Default to equal probabilities
                self.probabilities = [1.0 / len(self.strategies)] * len(self.strategies)
            if probabilities and len(probabilities) != len(strategies):
                raise ValueError(
                    "Number of probabilities must match number of strategies"
                )

    def apply_mask(self, batch: HeliosSample, **kwargs: Any) -> MaskedHeliosSample:
        """Apply multiple masking strategies to the input data.

        Args:
            batch: Input data of type HeliosSample
            **kwargs: Additional arguments for maskings

        Returns:
            MaskedHeliosSample containing the masked data and mask
        """
        # Choose a strategy - use probabilities if provided, otherwise use first strategy
        chosen_strategy = (
            random.choices(self.strategies, weights=self.probabilities, k=1)[0]
            if self.probabilities
            else self.strategies[0]
        )
        return chosen_strategy.apply_mask(batch, **kwargs)


@dataclass
class MaskingConfig(Config):
    """Configuration for masking strategies.

    Args:
        strategies: List of masking strategies to combine in the format of
        {
            "type": "random",
            # rest of init kwargs
        }
        probabilities: Optional list of probabilities for each strategy.
                     If None, strategies are applied sequentially.
    """

    strategies: list[dict[str, Any]]
    probabilities: list[float] | None = None

    def validate(self) -> None:
        """Validate the masking configuration."""
        if self.probabilities and len(self.probabilities) != len(self.strategies):
            raise ValueError("Number of probabilities must match number of strategies")

    def build(self) -> CompositeMaskingStrategy:
        """Build a CompositeMaskingStrategy from the config."""
        built_strategies = []
        for strategy_config in self.strategies:
            strategy_type = strategy_config.pop("type")
            strategy = MASKING_STRATEGY_REGISTRY[strategy_type](**strategy_config)
            built_strategies.append(strategy)

        return CompositeMaskingStrategy(
            strategies=built_strategies, probabilities=self.probabilities
        )
