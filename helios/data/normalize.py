"""Normalizer for the Helios dataset."""

import json
import random
from enum import Enum
from typing import Any

import numpy as np
from tqdm import tqdm

from helios.data.constants import IMAGE_TILE_SIZE, Modality, ModalitySpec
from helios.data.dataset import GetItemArgs, HeliosDataset
from helios.data.utils import update_streaming_stats


class Strategy(Enum):
    """The strategy to use for normalization."""

    # Whether to use predefined or computed values for normalization
    PREDEFINED = "predefined"
    COMPUTED = "computed"


class Normalizer:
    """Normalize the data."""

    def __init__(
        self,
        strategy: Strategy,
        std_multiplier: float | None = 2,
    ) -> None:
        """Initialize the normalizer.

        Args:
            strategy: The strategy to use for normalization (predefined or computed).
            std_multiplier: Optional, only for strategy COMPUTED.
                            The multiplier for the standard deviation when using computed values.

        Returns:
            None
        """
        self.strategy = strategy
        self.std_multiplier = std_multiplier
        self.norm_config = self._load_config()

    def _load_config(self) -> dict:
        """Load the appropriate config based on the modality strategy."""
        if self.strategy == Strategy.PREDEFINED:
            return self._load_predefined_config()
        elif self.strategy == Strategy.COMPUTED:
            return self._load_computed_config()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def _load_predefined_config(self) -> dict:
        """Load the predefined config."""
        with open("data/norm_configs/predefined.json") as f:
            return json.load(f)

    def _load_computed_config(self) -> dict:
        """Load the computed config."""
        with open("data/norm_configs/computed.json") as f:
            return json.load(f)

    def _normalize_predefined(
        self, modality: ModalitySpec, data: np.ndarray
    ) -> np.ndarray:
        """Normalize the data using predefined values."""
        # When using predefined values, we have the min and max values for each band
        modality_bands = modality.band_order
        modality_norm_values = self.norm_config[modality.name]
        min_vals = []
        max_vals = []
        for band in modality_bands:
            if band not in modality_norm_values:
                raise ValueError(f"Band {band} not found in config")
            min_val = modality_norm_values[band]["min"]
            max_val = modality_norm_values[band]["max"]
            min_vals.append(min_val)
            max_vals.append(max_val)
        # The last dimension of data is always the number of bands (channels)
        return (data - np.array(min_vals)) / (np.array(max_vals) - np.array(min_vals))

    def _normalize_computed(
        self, modality: ModalitySpec, data: np.ndarray
    ) -> np.ndarray:
        """Normalize the data using computed values."""
        # When using computed values, we compute the mean and std of each band in advance
        # Then convert the values to min and max values that cover ~90% of the data
        modality_bands = modality.band_order
        modality_norm_values = self.norm_config[modality.name]
        mean_vals = []
        std_vals = []
        for band in modality_bands:
            if band not in modality_norm_values:
                raise ValueError(f"Band {band} not found in config")
            mean_val = modality_norm_values[band]["mean"]
            std_val = modality_norm_values[band]["std"]
            mean_vals.append(mean_val)
            std_vals.append(std_val)
        min_vals = np.array(mean_vals) - self.std_multiplier * np.array(std_vals)
        max_vals = np.array(mean_vals) + self.std_multiplier * np.array(std_vals)
        return (data - min_vals) / (max_vals - min_vals)  # type: ignore

    def normalize(self, modality: ModalitySpec, data: np.ndarray) -> np.ndarray:
        """Normalize the data.

        Args:
            modality: The modality to normalize.
            data: The data to normalize.

        Returns:
            The normalized data.
        """
        if self.strategy == Strategy.PREDEFINED:
            return self._normalize_predefined(modality, data)
        elif self.strategy == Strategy.COMPUTED:
            return self._normalize_computed(modality, data)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")


def compute_normalization_values(
    dataset: HeliosDataset,
    estimate_from: int | None = None,
) -> dict[str, Any]:
    """Compute the normalization values for the dataset in a streaming manner.

    Args:
        dataset: The dataset to compute the normalization values for.
        estimate_from: The number of samples to estimate the normalization values from.

    Returns:
        dict: A dictionary containing the normalization values for the dataset.
    """
    dataset_len = len(dataset)
    if estimate_from is not None:
        indices_to_sample = random.sample(list(range(dataset_len)), k=estimate_from)
    else:
        indices_to_sample = list(range(dataset_len))

    norm_dict: dict[str, Any] = {}

    for i in tqdm(indices_to_sample):
        get_item_args = GetItemArgs(idx=i, patch_size=1, sampled_hw_p=IMAGE_TILE_SIZE)
        _, sample = dataset[get_item_args]
        for modality in sample.modalities:
            # Shall we compute the norm stats for worldcover?
            if modality == "timestamps" or modality == "latlon":
                continue
            modality_data = sample.as_dict(ignore_nones=True)[modality]
            modality_spec = Modality.get(modality)
            modality_bands = modality_spec.band_order
            if modality_data is None:
                continue
            if modality not in norm_dict:
                norm_dict[modality] = {}
                for band in modality_bands:
                    norm_dict[modality][band] = {
                        "mean": 0.0,
                        "var": 0.0,
                        "std": 0.0,
                        "count": 0,
                    }
            # Compute the normalization stats for the modality
            for idx, band in enumerate(modality_bands):
                modality_band_data = modality_data[:, :, :, idx]  # (H, W, T, C)
                current_stats = norm_dict[modality][band]
                new_count, new_mean, new_var = update_streaming_stats(
                    current_stats["count"],
                    current_stats["mean"],
                    current_stats["var"],
                    modality_band_data,
                )
                # Update the normalization stats
                norm_dict[modality][band]["count"] = new_count
                norm_dict[modality][band]["mean"] = new_mean
                norm_dict[modality][band]["var"] = new_var

    # Compute the standard deviation
    for modality in norm_dict:
        for band in norm_dict[modality]:
            norm_dict[modality][band]["std"] = (
                norm_dict[modality][band]["var"] / norm_dict[modality][band]["count"]
            ) ** 0.5

    norm_dict["total_n"] = dataset_len
    norm_dict["sampled_n"] = len(indices_to_sample)
    norm_dict["tile_path"] = dataset.tile_path

    return norm_dict
