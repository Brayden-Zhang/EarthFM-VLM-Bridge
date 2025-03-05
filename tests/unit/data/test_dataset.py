"""Unit tests for the dataset module."""

import pytest
import torch

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample, collate_helios_missing_modalities


@pytest.fixture
def samples_with_missing_modalities() -> list[HeliosSample]:
    """Samples with missing modalities."""
    s2_H, s2_W, s2_T, s2_C = 16, 16, 12, 13
    s1_H, s1_W, s1_T, s1_C = 16, 16, 12, 2
    wc_H, wc_W, wc_T, wc_C = 16, 16, 1, 10
    example_s2_data = torch.randn(s2_H, s2_W, s2_T, s2_C)
    example_s1_data = torch.randn(s1_H, s1_W, s1_T, s1_C)
    example_wc_data = torch.randn(wc_H, wc_W, wc_T, wc_C)
    example_latlon_data = torch.randn(2)
    timestamps = torch.tensor(
        [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.int32
    )

    sample1 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample2 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=None,
        worldcover=example_wc_data,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    sample_3 = HeliosSample(
        sentinel2_l2a=example_s2_data,
        sentinel1=example_s1_data,
        worldcover=None,
        latlon=example_latlon_data,
        timestamps=timestamps,
    )

    batch = [sample1, sample2, sample_3]
    return batch


def test_collate_helios_missing_modalities(
    samples_with_missing_modalities: list[HeliosSample],
):
    """Test the collate_helios_missing_modalities function."""

    collated_sample = collate_helios_missing_modalities(
        samples_with_missing_modalities,
        [
            Modality.SENTINEL2_L2A,
            Modality.SENTINEL1,
            Modality.WORLDCOVER,
            Modality.LATLON,
        ],
    )
    print(collated_sample.missing_modalities_masks)
    assert collated_sample.sentinel2_l2a.shape[0] == 3
    assert collated_sample.sentinel1.shape[0] == 3
    assert collated_sample.worldcover.shape[0] == 3
    assert collated_sample.latlon.shape[0] == 3
    assert collated_sample.timestamps.shape[0] == 3
    # Check missing modalities masks
    assert "sentinel1" in collated_sample.missing_modalities_masks
    assert "worldcover" in collated_sample.missing_modalities_masks
    assert "latlon" not in collated_sample.missing_modalities_masks
    assert "timestamps" not in collated_sample.missing_modalities_masks
    assert torch.equal(
        collated_sample.missing_modalities_masks["sentinel1"], torch.tensor([0, 1, 0])
    )
    assert torch.equal(
        collated_sample.missing_modalities_masks["worldcover"], torch.tensor([0, 0, 1])
    )


class TestHeliosSample:
    """Test the HeliosSample class."""

    # Test subsetting collate function with missing modalities
    def test_subset_with_missing_modalities(
        self,
        samples_with_missing_modalities: list[HeliosSample],
    ):
        collated_sample = collate_helios_missing_modalities(
            samples_with_missing_modalities,
            [
                Modality.SENTINEL2_L2A,
                Modality.SENTINEL1,
                Modality.WORLDCOVER,
            ],
        )
        hw_to_sample = list(range(4, 13))
        patch_size = 2
        max_tokens_per_instance = 100
        subset_sample = collated_sample.subset(
            patch_size=patch_size,
            max_tokens_per_instance=max_tokens_per_instance,
            hw_to_sample=hw_to_sample,
        )
        assert subset_sample.sentinel2_l2a.shape[0] == 3
        assert subset_sample.sentinel1.shape[0] == 3
        assert subset_sample.worldcover.shape[0] == 3

    # Next step will be to write the masking code
    # Then to modify the patch encoders
    # We don't want the missing tokens to be encluded in the token budget or in the encode decode ratio
    # lastly, to make the filtering modifications more configurable
