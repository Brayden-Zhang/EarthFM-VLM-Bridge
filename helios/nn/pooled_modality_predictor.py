""" Alternate Predictor that pools the tokens across modalities. And then predicts all the other modalities from that spatiall pooled representation"""

from helios.nn.flexihelios import Predictor, TokensAndMasks, return_modalities_from_dict, get_modalities_to_process
from helios.data.constants import Modality, ModalitySpec
from helios.train.masking import MaskedHeliosSample
import torch
from torch import Tensor
from olmo_core.config import Config
from dataclasses import dataclass
from helios.dataset.utils import get_modality_specs_from_names
import logging

logger = logging.getLogger(__name__)

# should this go after the composite encodings or before?
# It is only happening on the encoding tokens
# so after seems easier to implement because you otherwise need to repack everything to do this

# I should try both and see if there is a difference

# First I will do it after the composite encodings

class PooledModalityPredictor(Predictor):
    """Predictor that pools the tokens across modalities. And then predicts all the other modalities from that spatiall pooled representation"""


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
                data_list.append(tokens_dict[modality])
                mask_list.append(tokens_dict[masked_modality_name])
        return torch.cat(data_list, dim=1), torch.cat(mask_list, dim=1)

    def apply_attn(
        self,
        x: dict[str, Tensor],
        timestamps: Tensor,
        patch_size: int,
        input_res: int,
    ) -> dict[str, Tensor]:
        """Apply attention to the tokens."""
        tokens_only_dict, original_masks_dict, modalities_to_dims_dict = (
            self.split_tokens_masks_and_dims(x)
        )
        tokens_dict = self.composite_encodings(
            tokens_only_dict, timestamps, patch_size, input_res
        )
        tokens_dict.update(original_masks_dict)

        spatial_tokens, spatial_masks = self.stack_spatial_modalities_and_masks(tokens_dict)

        print(f"shape of spatial tokens: {spatial_tokens.shape}")
        print(f"shape of spatial masks: {spatial_masks.shape}")
        # Using this tokens dict we need to
        # Assume we are not using latlon
        # 1. Stack all the tokens together
        # 2. for each spatial-temporal location, I need to do masked attentive pooling to get a multi modality single channel vector
        # 3. Then I need to collapse these tokens to get the keys and values
        # 4. I can get the queries from the orignal decoder tokens
        # then i do cross attention
        # Then I can basically just repack this as I always have
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
                y=unmasked_tokens,
                attn_mask=(
                    unmasked_tokens_mask.bool() if not self.use_flash_attn else None
                ),  # only for flash attn though this should not be left in
                cu_seqlens_q=cu_seqlens_tokens_to_decode,
                cu_seqlens_k=cu_seqlens_unmasked_tokens,
                max_seqlen_q=max_length_of_tokens_to_decode,
                max_seqlen_k=max_length_of_unmasked_tokens,
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
        return Predictor(**kwargs)