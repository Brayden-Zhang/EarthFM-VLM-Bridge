"""Simple set up of latent predictor."""

# TODO: THis should be adapted when we make everything work with new architecture
from copy import deepcopy

import torch
import torch.nn as nn
from helios.nn.flexihelios import TokensAndMasks
from helios.train.masking import MaskedHeliosSample


class LatentMIMStyle(nn.Module):
    """Latent MIM Style."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        """Initialize the Latent MIM Style.

        Args:
            encoder: The encoder to use.
            decoder: The decoder to use.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_encoder = deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        x: MaskedHeliosSample,
        patch_size: int,
    ) -> TokensAndMasks:
        """Forward pass for the Latent MIM Style."""
        # TODO: Input And outputs here are not consistent between encoder and decoder need a tokensandmaks++
        latent = self.encoder(x, patch_size=patch_size)
        decoded = self.decoder(latent, timestamps=x.timestamps, patch_size=patch_size)
        return decoded
