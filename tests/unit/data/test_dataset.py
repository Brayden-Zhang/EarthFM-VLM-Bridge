"""Unit tests for the dataset module."""

import numpy as np
import pytest
import torch

from helios.data.constants import MISSING_VALUE, Modality
from helios.data.dataset import HeliosSample, collate_helios


@pytest.fixture
def samples_with_missing_modalities() -> list[HeliosSample]:
    """Samples with missing modalities."""
    s2_H, s2_W, s2_T, s2_C = 16, 16, 12, 13
    s1_H, s1_W, s1_T, s1_C = 16, 16, 12, 2
    wc_H, wc_W, wc_T, wc_C = 16, 16, 1, 10

    example_s2_data = np.random.randn(s2_H, s2_W, s2_T, s2_C)
    example_s1_data = np.random.randn(s1_H, s1_W, s1_T, s1_C)
    example_wc_data = np.random.randn(wc_H, wc_W, wc_T, wc_C)
    example_latlon_data = np.random.randn(2)
    timestamps = np.array([[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=np.int32)

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


def test_collate_helios(
    samples_with_missing_modalities: list[HeliosSample],
) -> None:
    """Test the collate_helios function."""
    collated_sample = collate_helios(
        samples_with_missing_modalities,
        [
            Modality.SENTINEL2_L2A,
            Modality.SENTINEL1,
            Modality.WORLDCOVER,
            Modality.LATLON,
        ],
    )

    # Check that all required fields are present
    assert collated_sample.sentinel2_l2a is not None
    assert collated_sample.sentinel1 is not None
    assert collated_sample.worldcover is not None
    assert collated_sample.latlon is not None
    assert collated_sample.timestamps is not None

    # Check the shapes
    assert collated_sample.sentinel2_l2a.shape[0] == 3
    assert collated_sample.sentinel1.shape[0] == 3
    assert collated_sample.worldcover.shape[0] == 3
    assert collated_sample.latlon.shape[0] == 3
    assert collated_sample.timestamps.shape[0] == 3

    # Check the missing modality mask values
    assert torch.all(collated_sample.sentinel1[1] == MISSING_VALUE)

    assert torch.all(collated_sample.worldcover[2] == MISSING_VALUE)


class TestHeliosSample:
    """Test the HeliosSample class."""

    # Test subsetting collate function with missing modalities
    def test_subset_with_missing_modalities(
        self,
        samples_with_missing_modalities: list[HeliosSample],
    ) -> None:
        """Test subsetting a collated sample with missing modalities."""
        collated_sample = collate_helios(
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

        # Check that the shapes are correct
        assert subset_sample.sentinel2_l2a is not None
        assert subset_sample.sentinel1 is not None
        assert subset_sample.worldcover is not None

        assert subset_sample.sentinel2_l2a.shape[0] == 3
        assert subset_sample.sentinel1.shape[0] == 3
        assert subset_sample.worldcover.shape[0] == 3

        # Check that the missing modality masks are preserved
        # Check the missing modality mask values
        assert torch.all(subset_sample.sentinel1[1] == MISSING_VALUE)
        assert torch.all(subset_sample.worldcover[2] == MISSING_VALUE)
