"""Model code for the Helios model."""

from typing import NamedTuple

import torch
from torch import Tensor, nn

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


class FlexiHeliosEmbeddings(nn.Module):
    """This will patchify and encode the data"""

    def __init__(
        self, list_of_modalities: list[str], max_patch_size: int, embedding_size: int
    ):
        """Initialize the embeddings"""
        super().__init__()
        self.list_of_modalities = list_of_modalities
        modality_to_bands_dict = {}
        self.per_modality_embeddings = nn.ModuleDict(
            {
                modality: FlexiPatchEmbed(
                    in_chans=len(self.modality_to_bands_dict[modality]),
                    embed_dim=embedding_size,
                    patch_size=max_patch_size,
                )
                for modality in list_of_modalities
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
        for modality in self.list_of_modalities:
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

        return output_dict


class FlexiPrestoBase(nn.Module):
    """Based on FlexiPrestoBase from presto-v3"""

    cross_attn: bool

    def __init__(
        self,
        embedding_size: int = 128,
        depth=2,
        mlp_ratio=2,
        num_heads=8,
        max_sequence_length=24,
        base_patch_size: int = 4,
        use_channel_embs: bool = True,
        drop_path: float = 0.0,
    ):
        super().__init__()

        self.space_time_groups = SPACE_TIME_BANDS_GROUPS_IDX
        self.space_groups = SPACE_BAND_GROUPS_IDX
        self.time_groups = TIME_BAND_GROUPS_IDX
        self.static_groups = STATIC_BAND_GROUPS_IDX
        self.embedding_size = embedding_size
        self.base_patch_size = base_patch_size
        self.max_sequence_length = max_sequence_length
        # we have 4 embeddings (pos_in_time, pos_in_space, month, channel) so each get
        # 0.25 of the dimension. This will change soon anyway
        self.pos_embed = nn.Parameter(
            get_1d_sincos_pos_embed_from_grid_torch(
                int(embedding_size * 0.25), torch.arange(max_sequence_length)
            ),
            requires_grad=False,
        )
        month_tab = get_month_encoding_table(int(embedding_size * 0.25))
        self.month_embed = nn.Embedding.from_pretrained(month_tab, freeze=True)
        if use_channel_embs:
            args = {"requires_grad": True}
        else:
            args = {"requires_grad": False}
        self.s_t_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_TIME_BANDS_GROUPS_IDX), int(embedding_size * 0.25)),
            **args,
        )
        self.sp_channel_embed = nn.Parameter(
            torch.zeros(len(SPACE_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.t_channel_embed = nn.Parameter(
            torch.zeros(len(TIME_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )
        self.st_channel_embed = nn.Parameter(
            torch.zeros(len(STATIC_BAND_GROUPS_IDX), int(embedding_size * 0.25)), **args
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @classmethod
    def collapse_and_combine_hwtc(
        cls,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
    ):
        s_t_x = rearrange(s_t_x, "b h w t c_g d -> b (h w t c_g) d")
        sp_x = rearrange(sp_x, "b h w c_g d -> b (h w c_g) d")
        t_x = rearrange(t_x, "b t c_g d -> b (t c_g) d")

        s_t_m = rearrange(s_t_m, "b h w t c_g-> b (h w t c_g)")
        sp_m = rearrange(sp_m, "b h w c_g-> b (h w c_g)")
        t_m = rearrange(t_m, "b t c_g -> b (t c_g)")

        x = torch.cat(
            [
                s_t_x,
                sp_x,
                t_x,
                st_x,
            ],
            dim=1,
        )
        m = torch.cat([s_t_m, sp_m, t_m, st_m], dim=1)
        return x, m

    @classmethod
    def split_and_expand_hwtc(
        cls,
        x: torch.Tensor,
        h: int,
        w: int,
        t: int,
        s_t_c_g: int,
        sp_c_g: int,
        t_c_g: int,
        st_c_g: int,
    ):
        n_s_t_t = h * w * t * s_t_c_g
        n_t_t = t * t_c_g

        s_t_x = rearrange(
            x[:, :n_s_t_t], "b (h w t c) d -> b h w t c d", h=h, w=w, t=t, c=s_t_c_g
        )
        sp_x = rearrange(
            x[:, n_s_t_t : -(n_t_t + st_c_g)],
            "b (h w c) d -> b h w c d",
            h=h,
            w=w,
            c=sp_c_g,
        )
        t_x = rearrange(
            x[:, -(n_t_t + st_c_g) : -st_c_g], "b (t c) d -> b t c d", t=t, c=t_c_g
        )
        st_x = x[:, -st_c_g:]

        return s_t_x, sp_x, t_x, st_x

    def apply_encodings(self, s_t_x, sp_x, t_x, st_x, months, patch_size, input_res):
        b, h, w, t, s_t_c_g, _ = s_t_x.shape
        sp_c_g, t_c_g = sp_x.shape[-2], t_x.shape[-2]
        st_c_g = st_x.shape[-2]

        s_t_channel = repeat(
            self.s_t_channel_embed, "c_g d -> b h w t c_g d", b=b, h=h, w=w, t=t
        )
        t_channel = repeat(self.t_channel_embed, "c_g d -> b t c_g d", b=b, t=t)
        st_channel = repeat(self.st_channel_embed, "c_g d -> b c_g d", b=b)
        sp_channel = repeat(
            self.sp_channel_embed, "c_g d -> b h w c_g d", b=b, h=h, w=w
        )

        pos_embed_s_t = repeat(
            self.pos_embed[:t], "t d -> b h w t c_g d", b=b, h=h, w=w, c_g=s_t_c_g
        )
        m_embed_s_t = repeat(
            self.month_embed(months), "b t d -> b h w t c_g d", h=h, w=w, c_g=s_t_c_g
        )

        pos_embed_t = repeat(self.pos_embed[:t], "t d -> b t c_g d", b=b, c_g=t_c_g)
        m_embed_t = repeat(self.month_embed(months), "b t d -> b t c_g d", c_g=t_c_g)
        t_zeros = torch.zeros(
            b, t, t_c_g, int(self.embedding_size * 0.25), device=t_x.device
        )

        sp_zeros = torch.zeros(
            b,
            h,
            w,
            sp_c_g,
            sp_channel.shape[-1] * 2,
            device=sp_channel.device,
        )

        st_zeros = torch.zeros(
            b, st_c_g, st_channel.shape[-1] * 3, device=st_channel.device
        )

        # find the resolution that each token represents, which will be
        # the number of pixels in a patch * the resolution of each pixel
        if patch_size is None:
            patch_size = self.base_patch_size
        token_res = input_res * patch_size
        gsd_ratio = token_res / BASE_GSD

        assert (
            h == w
        ), "get_2d_sincos_pos_embed_with_resolution currently requires that h==w"
        spatial_embed = get_2d_sincos_pos_embed_with_resolution(
            int(self.embedding_size * 0.25),
            h,
            torch.ones(b).to(s_t_x.device) * gsd_ratio,
            device=s_t_x.device,
        )
        spatial_embed = rearrange(spatial_embed, "b (h w) d -> b h w d", h=h, w=w)
        spatial_embed_s_t = repeat(
            spatial_embed, "b h w d -> b h w t c_g d", h=h, w=w, t=t, c_g=s_t_c_g
        )
        spatial_embed_s = repeat(
            spatial_embed, "b h w d -> b h w c_g d", h=h, w=w, c_g=sp_c_g
        )

        s_t_embed = torch.cat(
            [s_t_channel, pos_embed_s_t, m_embed_s_t, spatial_embed_s_t], dim=-1
        )
        sp_embed = torch.cat([sp_channel, sp_zeros, spatial_embed_s], dim=-1)
        t_embed = torch.cat([t_channel, pos_embed_t, m_embed_t, t_zeros], dim=-1)
        st_embed = torch.cat([st_channel, st_zeros], dim=-1)
        return s_t_x + s_t_embed, sp_x + sp_embed, t_x + t_embed, st_x + st_embed

    def apply_linear_projection(
        self,
        s_t_x: torch.Tensor,
        sp_x: torch.Tensor,
        t_x: torch.Tensor,
        st_x: torch.Tensor,
        s_t_m: torch.Tensor,
        sp_m: torch.Tensor,
        t_m: torch.Tensor,
        st_m: torch.Tensor,
        patch_size: int,
    ):
        """Given a [B, H, W, (T), C] inputs, returns a [B, H, W, (T), C_G, D] output.
        We assume that the spatial masks are consistent for the given patch size,
        so that if patch_size == 2 then one possible mask would be
        [0, 0, 1, 1]
        [0, 0, 1, 1]
        [1, 1, 0, 0]
        [1, 1, 0, 0]
        for the H, W dimensions
        """
        b, h, w, t, _ = s_t_x.shape
        new_h, new_w = h // patch_size, w // patch_size

        s_t_l, sp_l, t_l, st_l, s_t_m_l, sp_m_l, t_m_l, st_m_l = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for idx, (channel_group, channel_idxs) in enumerate(
            self.space_time_groups.items()
        ):
            s_t_m_l.append(s_t_m[:, 0::patch_size, 0::patch_size, :, idx])
            if s_t_m_l[-1].min() == 0:
                s_t_l.append(
                    self.space_time_embed[channel_group](
                        s_t_x[:, :, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                s_t_l.append(
                    torch.empty(
                        b,
                        new_h,
                        new_w,
                        t,
                        self.embedding_size,
                        dtype=s_t_x.dtype,
                        device=s_t_x.device,
                    )
                )
        for idx, (channel_group, channel_idxs) in enumerate(self.space_groups.items()):
            sp_m_l.append(sp_m[:, 0::patch_size, 0::patch_size, idx])
            if sp_m_l[-1].min() == 0:
                sp_l.append(
                    self.space_embed[channel_group](
                        sp_x[:, :, :, channel_idxs], patch_size=patch_size
                    )
                )
            else:
                sp_l.append(
                    torch.empty(
                        b,
                        new_h,
                        new_w,
                        self.embedding_size,
                        dtype=sp_x.dtype,
                        device=sp_x.device,
                    )
                )

        for idx, (channel_group, channel_idxs) in enumerate(self.time_groups.items()):
            t_m_l.append(t_m[:, :, idx])
            if t_m_l[-1].min() == 0:
                t_l.append(self.time_embed[channel_group](t_x[:, :, channel_idxs]))
            else:
                t_l.append(
                    torch.empty(
                        b, t, self.embedding_size, dtype=t_x.dtype, device=t_x.device
                    )
                )

        for idx, (channel_group, channel_idxs) in enumerate(self.static_groups.items()):
            st_m_l.append(st_m[:, idx])
            if st_m_l[-1].min() == 0:
                st_l.append(self.static_embed[channel_group](st_x[:, channel_idxs]))
            else:
                st_l.append(
                    torch.empty(
                        b, self.embedding_size, dtype=st_x.dtype, device=st_x.device
                    )
                )

        return (
            torch.stack(s_t_l, dim=-2),
            torch.stack(sp_l, dim=-2),
            torch.stack(t_l, dim=-2),
            torch.stack(st_l, dim=-2),
            torch.stack(s_t_m_l, dim=-1),
            torch.stack(sp_m_l, dim=-1),
            torch.stack(t_m_l, dim=-1),
            torch.stack(st_m_l, dim=-1),
        )


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
