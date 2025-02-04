"""Model code for the Helios model."""

from typing import NamedTuple

import torch
from einops import repeat
from torch import Tensor, nn

from helios.constants import BASE_GSD
from helios.nn.encodings import (
    get_1d_sincos_pos_encoding,
    get_2d_sincos_pos_encoding_with_resolution,
    get_month_encoding_table,
)
from helios.nn.flexi_patch_embed import FlexiPatchEmbed
from helios.train.masking import MaskedHeliosSample


class TokensAndMasks(NamedTuple):
    """Output to compute the loss on.

    Args:
        s2: sentinel 2 data of shape (B, C_G, T, P_H, P_W)
        s2_mask: sentinel 2 mask indicating which tokens are masked/unmasked
        latlon: lat lon data containing geographical coordinates
        latlon_mask: lat lon mask indicating which coordinates are masked/unmasked
    """

    s2: Tensor  # (B, C_G, T, P_H, P_W)
    s2_mask: Tensor
    latlon: Tensor
    latlon_mask: Tensor

    @property
    def device(self) -> torch.device:
        """Get the device of the tokens and masks."""
        return self.s2.device


class FlexiHeliosPatchEmbeddings(nn.Module):
    """This will patchify and encode the data"""

    def __init__(
        self,
        modalities_to_bands_dict: dict[str, list[int]],
        max_patch_size: int,
        embedding_size: int,
    ):
        """Initialize the embeddings"""
        super().__init__()
        self.modalities_to_bands_dict = modalities_to_bands_dict
        # WE want to be able to remove certain bands and moda
        self.per_modality_embeddings = nn.ModuleDict(
            {
                modality: FlexiPatchEmbed(
                    in_chans=len(bands),
                    embed_dim=embedding_size,
                    patch_size=max_patch_size,
                )
                for modality, bands in self.modalities_to_bands_dict.items()
            }
        )

    def get_masked_modality_name(self, modality: str) -> str:
        """Get the masked modality name."""
        return MaskedHeliosSample.get_masked_modality_name(modality)

    @staticmethod
    def is_any_data_seen_by_encoder(modality_mask: Tensor) -> bool:
        """Check if any data is seen by the encoder."""
        return modality_mask.min() == 0

    def apply_linear_projection(
        self,
        input_data: MaskedHeliosSample,
        patch_size: int,
    ):
        """Returns flexibly patchified embeddings for each modality of the input data

        Given a [B, H, W, (T), C] inputs, returns a [B, H, W, (T), C_G, D] output.
        We assume that the spatial masks are consistent for the given patch size,
        so that if patch_size == 2 then one possible mask would be
        [0, 0, 1, 1]
        [0, 0, 1, 1]
        [1, 1, 0, 0]
        [1, 1, 0, 0]
        for the H, W dimensions
        """
        # Calculate the new dimensions after patchification
        height = input_data.height
        width = input_data.width
        # perhaps return a dictionary instead of an un-named tuple
        new_height, new_width = height // patch_size, width // patch_size

        output_dict = {}
        for modality in self.modalities_to_bands_dict.keys():
            masked_modality_name = self.get_masked_modality_name(modality)
            modality_mask = getattr(input_data, masked_modality_name)
            # patchify masked data
            patchified_mask = modality_mask[:, 0::patch_size, 0::patch_size, :, :]
            output_dict[masked_modality_name] = patchified_mask

            if self.is_any_data_seen_by_encoder(modality_mask):
                modality_data = getattr(input_data, modality)
                patchified_data = self.per_modality_embeddings[modality](
                    modality_data, patch_size=patch_size
                )
            else:
                # If all data should be ignored by encoder, we need to return an empty tensor
                patchified_data = torch.empty(
                    modality_data.shape[0],
                    new_height,
                    new_width,
                    self.per_modality_embeddings[modality].embedding_size,
                    dtype=modality_data.dtype,
                    device=modality_data.device,
                )
            output_dict[modality] = patchified_data

        # TODO: IF possible we should be able to rewrap this intoa named tuple depends how we plan on using the time stuff at this point
        return MaskedHeliosSample(**output_dict)


class TokensOnly(NamedTuple):
    s2: torch.Tensor


# SHould this be called FlexiHeliosCompositeEncodings? or FlexiHeliosCompositeEmbeddings?
class FlexiHeliosCompositeEncodings(nn.Module):
    """This will apply the encodings to the patchified data"""

    def __init__(
        self,
        embedding_size: int,
        modalities_to_bands_dict: dict[str, list[int]],
        max_sequence_length: int,
        base_patch_size: int,
        use_channel_embs: bool = True,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.modalities_to_bands_dict = modalities_to_bands_dict
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size
        self.max_sequence_length = (
            max_sequence_length  # This max sequence length is a time dim thing
        )
        # we have 4 embeddings types (pos_in_time, pos_in_space, month, channel) so each get
        # 0.25 of the dimension
        embedding_dim_per_embedding_type = embedding_size * 0.25
        # Position encodings for time dimension initialized to 1D sinusoidal encodings
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_encoding(
                torch.arange(max_sequence_length),
                embedding_dim_per_embedding_type,
            ),
            requires_grad=False,
        )
        # M
        month_tab = get_month_encoding_table(embedding_dim_per_embedding_type)
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        if use_channel_embs:
            args = {"requires_grad": True}
        else:
            args = {"requires_grad": False}

        self.per_modality_channel_embeddings = nn.ModuleDict(
            {
                modality: nn.Parameter(
                    torch.zeros(len(bands), embedding_dim_per_embedding_type),
                    **args,
                )
                for modality, bands in self.modalities_to_bands_dict.items()
            }
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @property
    def device(self) -> torch.device:
        """Get the device of the tokens and masks."""
        return self.s2.device

    @staticmethod
    def calculate_gsd_ratio(input_res: float, patch_size: int) -> float:
        """Calculate the Ground Sample Distance ratio."""
        return input_res * patch_size / BASE_GSD

    def apply_encodings(
        self,
        per_modality_input_tokens: TokensOnly,
        months: Tensor,  # Shouldn't this vary per modality
        patch_size: int,  # does this vary per modality
        input_res: Tensor,  # Does this vary per modality
    ) -> TokensOnly:
        """Apply the encodings to the patchified data"""
        # We need a test that keeps all of this organized so that we can easily add new modalities
        # There shoud be a named tuple isntead of a dict here
        # How do we handle missing modalities? We are assuming that by this point we have already padded
        # DO we need  to support Dropping modalities entirely? Probably
        # and masked the data so that we have a consistent shape
        output_dict = {}
        for modality in self.modalities_to_bands_dict.keys():
            # TODO: We will need to be able to handle modalities that do not need all these types of encodings
            # For right now we are going to have S1, S2 and worldcover so this does not support worldcover
            modality_tokens: Tensor = getattr(per_modality_input_tokens, modality)
            if len(modality_tokens.shape) != 5:
                raise NotImplementedError(
                    "Only modalities that have space time, width, height dims are supported"
                )
            b, h, w, t, c_g = modality_tokens.shape

            modality_channel_embed = self.per_modality_channel_embeddings[modality]
            modality_channel_embed = repeat(
                modality_channel_embed, "c_g d -> b h w c_g d", b=b, h=h, w=w
            )

            # Create time position encodings and month encodings for each modality (maybe we should have just an overall yealry encoding?)
            modality_pos_embed = repeat(
                self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=c_g
            )
            # Are the months the same for all the modalities?
            modality_month_embed = repeat(
                self.month_embed(months), "b t d -> b h w t c_g d", h=h, w=w, c_g=c_g
            )

            # Pad the embeddings if one of the embedding types is not applicable for a given modality

            # find the resolution that each token represents, which will be
            # the number of pixels in a patch * the resolution of each pixel

            gsd_ratio = self.calculate_gsd_ratio(input_res, patch_size)

            # We also want a 2D space
            assert (
                h == w
            ), "get_2d_sincos_pos_embed_with_resolution currently requires that h==w"
            current_device = modality_tokens.device
            spatial_embed = get_2d_sincos_pos_encoding_with_resolution(
                grid_size=h,
                res=torch.ones(b, device=current_device) * gsd_ratio,
                encoding_dim=int(self.embedding_size * 0.25),
                device=current_device,
            )
            modality_embed = torch.cat(
                [
                    modality_channel_embed,
                    modality_pos_embed,
                    modality_month_embed,
                    spatial_embed,
                ],
                dim=-1,
            )
            output_dict[modality] = modality_embed + modality_tokens

        return TokensOnly(**output_dict)


# class FlexiPrestoBase(nn.Module):
#     """Based on FlexiPrestoBase from presto-v3"""

#     cross_attn: bool

#     def __init__(
#         self,
#         embedding_size: int = 128,
#         depth=2,
#         mlp_ratio=2,
#         num_heads=8,
#         max_sequence_length=24,
#         base_patch_size: int = 4,
#         use_channel_embs: bool = True,
#         drop_path: float = 0.0,
#     ):
#         super().__init__()
#         self.embedding_size = embedding_size
#         self.base_patch_size = base_patch_size
#         self.max_sequence_length = max_sequence_length
#         # we have 4 embeddings (pos_in_time, pos_in_space, month, channel) so each get
#         # 0.25 of the dimension. This will change soon anyway

#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             # we use xavier_uniform following official JAX ViT:
#             torch.nn.init.xavier_uniform_(m.weight)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)

#     @classmethod
#     def collapse_and_combine_hwtc(
#         cls,
#         s_t_x: torch.Tensor,
#         sp_x: torch.Tensor,
#         t_x: torch.Tensor,
#         st_x: torch.Tensor,
#         s_t_m: torch.Tensor,
#         sp_m: torch.Tensor,
#         t_m: torch.Tensor,
#         st_m: torch.Tensor,
#     ):
#         s_t_x = rearrange(s_t_x, "b h w t c_g d -> b (h w t c_g) d")
#         sp_x = rearrange(sp_x, "b h w c_g d -> b (h w c_g) d")
#         t_x = rearrange(t_x, "b t c_g d -> b (t c_g) d")

#         s_t_m = rearrange(s_t_m, "b h w t c_g-> b (h w t c_g)")
#         sp_m = rearrange(sp_m, "b h w c_g-> b (h w c_g)")
#         t_m = rearrange(t_m, "b t c_g -> b (t c_g)")

#         x = torch.cat(
#             [
#                 s_t_x,
#                 sp_x,
#                 t_x,
#                 st_x,
#             ],
#             dim=1,
#         )
#         m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=1)
#         return x, m

#     @classmethod
#     def split_and_expand_hwtc(
#         cls,
#         x: torch.Tensor,
#         h: int,
#         w: int,
#         t: int,
#         s_t_c_g: int,
#         sp_c_g: int,
#         t_c_g: int,
#         st_c_g: int,
#     ):
#         n_s_t_t = h * w * t * s_t_c_g
#         n_t_t = t * t_c_g

#         s_t_x = rearrange(
#             x[:, :n_s_t_t], "b (h w t c) d -> b h w t c d", h=h, w=w, t=t, c=s_t_c_g
#         )
#         sp_x = rearrange(
#             x[:, n_s_t_t : -(n_t_t + st_c_g)],
#             "b (h w c) d -> b h w c d",
#             h=h,
#             w=w,
#             c=sp_c_g,
#         )
#         t_x = rearrange(
#             x[:, -(n_t_t + st_c_g) : -st_c_g], "b (t c) d -> b t c d", t=t, c=t_c_g
#         )
#         st_x = x[:, -st_c_g:]

#         return s_t_x, sp_x, t_x, st_x


# I want this class to be slighlty more agnostic to the passed in encoding class and have that be configurable too
class Encoder(nn.Module):
    """Encoder module that processes masked input samples into token representations."""

    # apply linear input projection
    # apply Encodings
    # Apply attention
    # apply Norm

    def forward(self, x: MaskedHeliosSample, patch_size: int) -> TokensAndMasks:
        """Process masked input samples into token representations.

        Args:
            x: Masked input sample containing the data to be encoded
            patch_size: Size of patches to divide the input into

        Returns:
            TokensAndMasks containing the encoded representations and their masks
        """
        raise NotImplementedError


class Predictor(nn.Module):
    """Predictor module that generates predictions from encoded tokens."""

    def forward(self, x: TokensAndMasks) -> TokensAndMasks:
        """Generate predictions from encoded token representations.

        Args:
            x: TokensAndMasks containing the encoded tokens to make predictions from

        Returns:
            TokensAndMasks containing the predicted tokens and their masks
        """
        raise NotImplementedError
