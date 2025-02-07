"""Test masking."""

import torch

from helios.data.dataset import HeliosSample
from helios.train.masking import MaskValue, RandomMaskingStrategy


def test_random_masking() -> None:
    """Test random masking ratios."""
    b, h, w, t = 100, 16, 16, 8

    modalities_dict = dict(
        {"s2": dict({"rgb": [0, 1, 2], "nir": [3]}), "latlon": dict({"latlon": [0, 1]})}
    )

    days = torch.randint(1, 31, (b, 1, t), dtype=torch.long)
    months = torch.randint(1, 13, (b, 1, t), dtype=torch.long)
    years = torch.randint(2018, 2020, (b, 1, t), dtype=torch.long)
    timestamps = torch.cat([days, months, years], dim=1)  # Shape: (B, 3, T)

    batch = HeliosSample(
        s2=torch.ones((b, 4, t, h, w)), latlon=torch.ones((b, 2)), timestamps=timestamps
    )
    encode_ratio, decode_ratio = 0.25, 0.5
    masked_sample = RandomMaskingStrategy().apply_mask(
        batch,
        patch_size=4,
        modalities_to_channel_groups_dict=modalities_dict,
        encode_ratio=encode_ratio,
        decode_ratio=decode_ratio,
    )

    # check that each modality has the right masking ratio
    for modality_name in masked_sample._fields:
        if modality_name.endswith("mask"):
            mask = getattr(masked_sample, modality_name)
            total_elements = mask.numel()
            num_encoder = len(mask[mask == MaskValue.ONLINE_ENCODER.value])
            num_decoder = len(mask[mask == MaskValue.DECODER_ONLY.value])
            assert (
                num_encoder / total_elements
            ) == encode_ratio, f"{modality_name} has incorrect encode mask ratio"
            assert (
                num_decoder / total_elements
            ) == decode_ratio, f"{modality_name} has incorrect decode mask ratio"
