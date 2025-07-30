""" Alternate Predictor that pools the tokens across modalities. And then predicts all the other modalities from that spatiall pooled representation"""

from typing import Any
from helios.nn.flexihelios import Predictor, TokensAndMasks, return_modalities_from_dict, get_modalities_to_process
from helios.data.constants import Modality, ModalitySpec
from helios.train.masking import MaskedHeliosSample, MaskValue
import torch
from torch import Tensor
from olmo_core.config import Config
from dataclasses import dataclass
from helios.dataset.utils import get_modality_specs_from_names
import logging
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import fully_shard
logger = logging.getLogger(__name__)

# should this go after the composite encodings or before?
# It is only happening on the encoding tokens
# so after seems easier to implement because you otherwise need to repack everything to do this

# I should try both and see if there is a difference

# First I will do it after the composite encodings


class AttnPool(nn.Module):
    """Attention Pooling Probe.

    Args:
        in_dim (int): Input feature dimension. Must be divisible by 64.
        out_dim (int): Output dimension (typically num_classes * patch_size * patch_size).

    Attributes:
        query_token (nn.Parameter): Learnable query token for attention pooling.
        num_heads (int): Number of attention heads.
        kv (nn.Linear): Linear layer to produce keys and values.
        linear (nn.Linear): Final linear layer for output logits.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the attention pooling linear probe."""
        super().__init__()
        assert in_dim % 64 == 0, "in_dim must be divisible by 64"
        self.query_token: nn.Parameter = nn.Parameter(torch.empty(in_dim))
        self.num_heads: int = in_dim // 64
        self.kv: nn.Linear = nn.Linear(in_dim, in_dim * 2)
        self.linear: nn.Linear = nn.Linear(in_dim, out_dim)
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize weights for the probe."""
        nn.init.trunc_normal_(self.query_token, std=0.02)
        nn.init.trunc_normal_(self.kv.weight, std=0.02)
        nn.init.zeros_(self.kv.bias)
        nn.init.trunc_normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, feat_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for attention pooling linear probe.

        Args:
            feat_tokens (torch.Tensor): Input feature tokens of shape (B, H, W, N, D).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Output logits after linear layer, shape (B, H, W, out_dim).
                - Attention weights, shape (B*H*W, num_heads, 1, N).
        """

        # B, H, W, N, D = feat_tokens.shape
        # feat_tokens = rearrange(feat_tokens, "b h w n d -> (b h w) n d")
        logger.info(f"shape of feat_tokens: {feat_tokens.shape}")
        collapsed_dim , N, D = feat_tokens.shape
        q = self.query_token.expand(collapsed_dim, 1, -1)
        q = q.reshape(
            collapsed_dim, 1, self.num_heads, D // self.num_heads
        )  # [B, 1, head, D_head]
        q = rearrange(q, "b h n d -> b n h d")
        # log the dtype of kv weights and the feat_tokens
        logger.info(f"dtype of kv weights: {self.kv.weight.dtype}")
        logger.info(f"dtype of feat_tokens: {feat_tokens.dtype}")
        # convert feat_tokens to dto
        # why is this hack needed?
        feat_tokens = feat_tokens.to(self.kv.weight.dtype)
        kv = self.kv(feat_tokens).reshape(
            collapsed_dim, N, 2, self.num_heads, D // self.num_heads
        )  # [B, N, 2, head, D_head]
        kv = rearrange(kv, "b n two h d -> two b h n d")
        k, v = torch.unbind(kv, dim=0)  # 2 * [B, head, N, D_head]
        logger.info(f"shape of k: {k.shape}")
        logger.info(f"shape of v: {v.shape}")
        logger.info(f"shape of q: {q.shape}")
        # mask shape
        logger.info(f"shape of mask: {mask.shape}")
        if mask is not None:
            mask = mask[:, None, None].repeat((1, self.num_heads, 1, 1))
        logger.info(f"shape of mask: {mask.shape}")

        # True indicates that the token should take part in attention
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # [B, head, 1, D_head]
        x = rearrange(x, "b h 1 d -> b (h d)")
        return x

class PooledModalityPredictor(Predictor):
    """Predictor that pools the tokens across modalities. And then predicts all the other modalities from that spatiall pooled representation"""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.attn_pool = AttnPool(self.embedding_size, self.embedding_size)

    def stack_spatial_modalities_and_masks(self, tokens_dict: dict[str, Tensor]) -> Tensor:
        """Stack the spatial modalities together."""
        available_modalities = return_modalities_from_dict(tokens_dict)
        modalities_to_process = get_modalities_to_process(
            available_modalities, self.supported_modality_names
        )
        mask_list = []
        data_list = []
        for modality in modalities_to_process:
            if Modality.get(modality).is_spatial:
                masked_modality_name = MaskedHeliosSample.get_masked_modality_name(modality)
                logger.info(f"modality: {modality}, masked_modality_name: {masked_modality_name}")
                data = tokens_dict[modality]
                mask = tokens_dict[masked_modality_name]
                logger.info(f"shape of data: {data.shape}")
                logger.info(f"shape of mask: {mask.shape}")
                data_list.append(data)
                mask_list.append(mask)
        # stack in the modality dimension
        return torch.cat(data_list, dim=4), torch.cat(mask_list, dim=4)

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> dict[str, Tensor]:
        """Apply attention to the tokens."""
        logger.warning(
            "Calling apply_attn for PooledModalityPredictor"
        )
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        tokens_dict.update(original_masks_dict)

        spatial_tokens, spatial_masks = self.stack_spatial_modalities_and_masks(tokens_dict)
        # I want to get to a shape of (B, H, W T) x M x D and then attentive pool across modalities
        B, H, W, T, M, D = spatial_tokens.shape
        spatial_tokens = rearrange(spatial_tokens, "b h w t m d -> (b h w t) m d")

        spatial_masks = rearrange(spatial_masks, "b h w t m -> (b h w t) m")
        # print the unique values of the masks
        logger.info(f"unique values of the masks: {torch.unique(spatial_masks)}")
        pooled_attn_mask = spatial_masks == MaskValue.ONLINE_ENCODER.value
        # Do I potentially need to filter out tokens that have no online marked modalities? Maybe not because we will just disgard those


        pooled_tokens = self.attn_pool(spatial_tokens, pooled_attn_mask)
        logger.info(f"shape of pooled tokens: {pooled_tokens.shape}")
        pooled_tokens = rearrange(pooled_tokens, "(b h w t) d -> b (h w t) d", b=B, h=H, w=W, t=T, d=D)
        # for spatial_masks if any in the modality dimension is online encode, set the token to online encoder only
        # otherwise set to Missing Value
        online_encoder_only_mask = (spatial_masks == MaskValue.ONLINE_ENCODER.value).any(dim=-1)
        pooled_attn_mask = torch.where(online_encoder_only_mask, MaskValue.ONLINE_ENCODER.value, MaskValue.MISSING.value)

        pooled_attn_mask = rearrange(pooled_attn_mask, "(b h w t) -> b (h w t)", b=B, h=H, w=W, t=T)
        logger.info(f"shape of pooled tokens: {pooled_tokens.shape}")

        (
            _,
            pooled_tokens,
            _,
            pooled_attn_mask,
            _,
            _,
            _,
            _,
            _,
        ) = self.split_x_y(pooled_tokens, pooled_attn_mask)


        # I need to do a step where I basically split the pooled tokens up so that I have an instance wide
        # collapsed mask of these

        all_tokens, mask = self.collapse_and_combine_hwtc(tokens_dict)
        # X contains the tokens to decode, Y contains the tokens to attend to for context
        (
            tokens_to_decode,
            unmasked_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            indices,
            seqlens_tokens_to_decode,
            seqlens_unmasked_tokens,
            max_length_of_tokens_to_decode,
            max_length_of_unmasked_tokens,
        ) = self.split_x_y(all_tokens, mask)
        # Pack x tokens
        if self.use_flash_attn:
            og_shape_tokens_to_decode = tokens_to_decode.shape
            tokens_to_decode = self.pack_tokens(
                tokens_to_decode, tokens_to_decode_mask.bool()
            )
            og_shape_unmasked_tokens = unmasked_tokens.shape
            unmasked_tokens = self.pack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool()
            )
            cu_seqlens_tokens_to_decode = get_cumulative_sequence_lengths(
                seqlens_tokens_to_decode
            )
            cu_seqlens_unmasked_tokens = get_cumulative_sequence_lengths(
                seqlens_unmasked_tokens
            )
        else:
            cu_seqlens_tokens_to_decode = None
            cu_seqlens_unmasked_tokens = None

        for blk in self.blocks:
            # note that we are not taking the inverse of the mask, since split_x_y gives us
            # true values for values we want to take part in attention
            tokens_to_decode = blk(
                x=tokens_to_decode,
                y=pooled_tokens,
                attn_mask=(
                    pooled_attn_mask.bool() if not self.use_flash_attn else None
                ),  # only for flash attn though this should not be left in
                # Assume not compatible with flash attn for now
                # cu_seqlens_q=cu_seqlens_tokens_to_decode,
                # cu_seqlens_k=cu_seqlens_unmasked_tokens,
                # max_seqlen_q=max_length_of_tokens_to_decode,
                # max_seqlen_k=max_length_of_unmasked_tokens,
            )

        if self.use_flash_attn:
            tokens_to_decode = self.unpack_tokens(
                tokens_to_decode,
                tokens_to_decode_mask.bool(),
                og_shape_tokens_to_decode,
            )
            unmasked_tokens = self.unpack_tokens(
                unmasked_tokens, unmasked_tokens_mask.bool(), og_shape_unmasked_tokens
            )

        x = self.combine_x_y(
            tokens_to_decode=tokens_to_decode,
            unmasked_tokens=unmasked_tokens,
            tokens_to_decode_mask=tokens_to_decode_mask,
            unmasked_tokens_mask=unmasked_tokens_mask,
            indices=indices,
        )
        tokens_per_modality_dict = self.split_and_expand_per_modality(
            x, modalities_to_dims_dict
        )
        tokens_per_modality_dict.update(original_masks_dict)
        return tokens_per_modality_dict

@dataclass
class PooledModalityPredictorConfig(Config):
    """Configuration for the Predictor."""

    supported_modality_names: list[str]
    encoder_embedding_size: int = 16
    decoder_embedding_size: int = 16
    depth: int = 2
    mlp_ratio: float = 1.0
    num_heads: int = 2
    max_sequence_length: int = 12
    drop_path: float = 0.0
    learnable_channel_embeddings: bool = True
    random_channel_embeddings: bool = False
    output_embedding_size: int | None = None
    use_flash_attn: bool = False
    qk_norm: bool = False

    def validate(self) -> None:
        """Validate the configuration."""
        if len(self.supported_modalities) == 0:
            raise ValueError("At least one modality must be added!")
        else:
            for modality in self.supported_modalities:
                if modality not in Modality.values():
                    raise ValueError(f"Modality {modality} is not supported")

    @property
    def supported_modalities(self) -> list[ModalitySpec]:
        """Get the supported modalities."""
        return get_modality_specs_from_names(self.supported_modality_names)

    def build(self) -> "Predictor":
        """Build the predictor."""
        self.validate()
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        # supported_modality_names is replaced by supported_modalities
        kwargs.pop("supported_modality_names")
        kwargs["supported_modalities"] = self.supported_modalities
        logger.info(f"Predictor kwargs: {kwargs}")
        return PooledModalityPredictor(**kwargs)