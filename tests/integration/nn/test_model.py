"""Integration tests for the model.

Any methods that piece together multiple steps or are the entire forward pass for a module should be here
"""

import pytest
import torch

from helios.nn.model import Encoder, Predictor, TokensAndMasks
from helios.train.masking import MaskValue


# TODO: We should have a loaded Test batch with real data for this one
class TestEncoder:
    """Integration tests for the Encoder class."""

    @pytest.fixture
    def encoder(self) -> Encoder:
        """Create encoder fixture for testing.

        Returns:
            Encoder: Test encoder instance with small test config
        """
        modalities_dict = dict({"s2": dict({"rgb": [0, 1, 2], "nir": [3]})})
        return Encoder(
            embedding_size=16,
            max_patch_size=8,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            modalities_to_channel_groups_dict=modalities_dict,
            max_sequence_length=12,
            base_patch_size=4,
            use_channel_embs=True,
        )

    def test_apply_attn(self, encoder: Encoder) -> None:
        """Test applying attention layers with masking via the apply_attn method.

        Args:
            encoder: Test encoder instance
        """
        s2_tokens = torch.randn(1, 2, 2, 3, 2, 16)
        s2_mask = torch.zeros(1, 2, 2, 3, 2, dtype=torch.long)

        # Mask the first and second "positions" in this 2x2 grid.
        s2_mask[0, 0, 0, 0] = 1  # mask first token
        s2_mask[0, 0, 1, 0] = 1  # mask second token

        # Construct the TokensAndMasks namedtuple with mock modality data + mask.
        x = TokensAndMasks(s2=s2_tokens, s2_mask=s2_mask)

        timestamps = (
            torch.tensor(
                [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
            )
            .unsqueeze(0)
            .permute(0, 2, 1)
        )  # [B, 3, T]
        patch_size = 4
        input_res = 10

        output = encoder.apply_attn(
            x=x, timestamps=timestamps, patch_size=patch_size, input_res=input_res
        )

        assert isinstance(
            output, TokensAndMasks
        ), "apply_attn should return a TokensAndMasks object."

        # Ensure shape is preserved in the output tokens.
        assert (
            output.s2.shape == s2_tokens.shape
        ), f"Expected output 's2' shape {s2_tokens.shape}, got {output.s2.shape}."

        # Confirm the mask was preserved and that masked tokens are zeroed out in the output.
        assert (output.s2_mask == s2_mask).all(), "Mask should be preserved in output"
        assert (
            output.s2[s2_mask >= MaskValue.TARGET_ENCODER_ONLY.value] == 0
        ).all(), "Masked tokens should be 0 in output"

    def test_forward(self, encoder: Encoder) -> None:
        """Test full forward pass.

        Args:
            encoder: Test encoder instance
        """
        pass


class TestPredictor:
    """Integration tests for the Predictor class."""

    @pytest.fixture
    def predictor(self) -> Predictor:
        """Create predictor fixture for testing.

        Returns:
            Predictor: Test predictor instance with small test config
        """
        """Create predictor fixture for testing."""
        modalities_to_channel_groups_dict = dict(
            {"s2": dict({"rgb": [0, 1, 2], "nir": [3]})}
        )
        return Predictor(
            modalities_to_channel_groups_dict=modalities_to_channel_groups_dict,
            encoder_embedding_size=8,
            decoder_embedding_size=16,
            depth=2,
            mlp_ratio=4.0,
            num_heads=2,
            max_sequence_length=12,
            max_patch_size=8,
            drop_path=0.1,
            learnable_channel_embeddings=True,
            output_embedding_size=8,
        )

    def test_predictor_forward(self, predictor: Predictor) -> None:
        """Test the full forward pass of the Predictor."""
        B = 1  # Batch size
        H = 2  # Spatial height
        W = 2  # Spatial width
        T = 3  # Number of timesteps
        num_groups = 2  # Number of channel groups (as defined in modalities_to_channel_groups_dict)

        embedding_dim = predictor.encoder_to_decoder_embed.in_features

        s2_tokens = torch.randn(B, H, W, T, num_groups, embedding_dim)

        s2_mask = torch.full(
            (B, H, W, T, num_groups),
            fill_value=MaskValue.DECODER_ONLY.value,
            dtype=torch.float32,
        )

        # Create dummy latitude and longitude data (and its mask)
        latlon = torch.randn(B, 2)
        latlon_mask = torch.ones(B, 2, dtype=torch.float32)

        encoded_tokens = TokensAndMasks(
            s2=s2_tokens, s2_mask=s2_mask, latlon=latlon, latlon_mask=latlon_mask
        )
        timestamps = torch.tensor(
            [[[1, 15, 30], [6, 7, 8], [2018, 2018, 2018]]],
            dtype=torch.long,
        )

        patch_size = 4
        input_res = 1

        output = predictor.forward(encoded_tokens, timestamps, patch_size, input_res)

        expected_token_shape = (B, H, W, T, num_groups, predictor.output_embedding_size)
        assert (
            output.s2.shape == expected_token_shape
        ), f"Expected tokens shape {expected_token_shape}, got {output.s2.shape}"

        expected_mask_shape = (B, H, W, T, num_groups)
        assert (
            output.s2_mask.shape == expected_mask_shape
        ), f"Expected mask shape {expected_mask_shape}, got {output.s2_mask.shape}"
