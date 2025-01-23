"""Collator functions for the dataset."""

import logging
from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class PerModalityCollatedOutput(NamedTuple):
    """Output for collation by modality."""

    sentinel2: torch.Tensor
    naip: torch.Tensor
    worldcover: torch.Tensor
    sample_metadata: list[dict]


def pad_time_dim(
    data: torch.Tensor, num_timesteps: int, max_len: int, pad_token_value: float = 0.0
) -> torch.Tensor:
    """Pad the time dimension of the data to the max length.

    Assumes the data is in the shape (h, w, t, c)
    """
    pad_shape = (0, 0, 0, int(max_len - num_timesteps), 0, 0)
    return F.pad(
        data,
        pad_shape,
        value=pad_token_value,
    )


def per_modality_collate_fn(items: Sequence[dict]) -> PerModalityCollatedOutput:
    """Collate function for inputs with variable time data into per modality tuples.

    Args:
        items: Sequence of dictionaries containing data for each modality

    Returns:
        PerModalityCollatedOutput containing batched tensors for each modality
    """
    max_len = max(item["num_timesteps"] for item in items)
    all_sentinel2 = []
    all_naip = []
    all_worldcover = []
    sample_metadata = []

    for item in items:
        data_inputs = item["data_inputs"]
        assert "sentinel2" in data_inputs.keys()
        assert "naip" in data_inputs.keys()
        assert "worldcover" in data_inputs.keys()

        all_sentinel2.append(
            pad_time_dim(
                torch.as_tensor(data_inputs["sentinel2"]),
                item["num_timesteps"],
                max_len,
            )
        )
        all_naip.append(torch.as_tensor(data_inputs["naip"]))
        all_worldcover.append(torch.as_tensor(data_inputs["worldcover"]))
        sample_metadata.append(item["sample_metadata"])
    sentinel2_batch = torch.stack(all_sentinel2)
    naip_batch = torch.stack(all_naip)
    worldcover_batch = torch.stack(all_worldcover)
    return PerModalityCollatedOutput(
        sentinel2=sentinel2_batch,
        naip=naip_batch,
        worldcover=worldcover_batch,
        sample_metadata=sample_metadata,
    )
