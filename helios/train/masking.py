"""Masking module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from class_registry import ClassRegistry
from olmo_core.config import Config

from helios.data.dataset import HeliosSample
from helios.types import ArrayTensor


class MaskValue(Enum):
    """Masks can take 3 possible values.

    ONLINE_ENCODER: The token is seen by the online encoder
    TARGET_ENCODER_ONLY: The token is seen by the target encoder only
    DECODER_ONLY: The token is seen by the decoder only
    """

    ONLINE_ENCODER = 0
    TARGET_ENCODER_ONLY = 1
    DECODER_ONLY = 2


# SHould be return type of masking strategy
class MaskedHeliosSample(NamedTuple):
    """A masked sample of the data from the Helios dataset.

    This is a namedtuple that contains the data for a single sample from the Helios dataset.
    latlon and timestamps are the same for all modalities.
    For each modality. we have an ArrayTensor named by modality, and a mask for each modality named by modality_mask.
    we also have a mask for the latlon called latlon_mask

    Args:
        s2: ArrayTensor  # [B, len(S2_bands), T H, W]
        s2_mask: ArrayTensor
        latlon: ArrayTensor  # [B, 2]
        latlon_mask: ArrayTensor
        timestamps: ArrayTensor  # [B, D=3, T], where D=[day, month, year]
    """

    s2: ArrayTensor  # [B, len(S2_bands), T H, W]
    s2_mask: ArrayTensor
    latlon: ArrayTensor  # [B, 2]
    latlon_mask: ArrayTensor
    timestamps: (
        ArrayTensor  # [B, D=3, T], where D=[day, month, year] (months are zero indexed)
    )

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


@dataclass
class MaskingConfig(Config):
    """Configuration for masking strategies.

    Args:
        strategy_config: Masking strategy to use in the format of
        {
            "type": "random", # registry key
            # rest of init kwargs
        }
    """

    strategy_config: dict[str, Any]

    def build(self) -> type[MaskingStrategy]:
        """Build a MaskingStrategy from the config."""
        return MASKING_STRATEGY_REGISTRY[self.strategy_config["type"]](
            **self.strategy_config
        )
