"""Embeddings from models."""

import logging

import torch
from torch.utils.data import DataLoader

from helios.nn.flexihelios import Encoder

logger = logging.getLogger(__name__)


def get_embeddings(
    data_loader: DataLoader, model: Encoder
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get embeddings from model for the data in data_loader."""
    embeddings = []
    labels = []

    model = model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for masked_helios_sample, label in data_loader:
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                # TODO: Model expects masked helios sample we need to pass empty masks
                # Likely we want to have a flag that checks for eval mode and passes empty masks
                batch_embeddings = model(
                    masked_helios_sample, patch_size=model.base_patch_size
                )  # (bsz, dim)

            embeddings.append(batch_embeddings.to(torch.bfloat16).cpu())
            labels.append(label)

    embeddings = torch.cat(embeddings, dim=0)  # (N, dim)
    labels = torch.cat(labels, dim=0)  # (N)

    return embeddings, labels
