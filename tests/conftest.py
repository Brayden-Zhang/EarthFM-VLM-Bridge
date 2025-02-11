"""Conftest for the tests."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def set_random_seeds() -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)


@pytest.fixture
def modalities_to_channel_groups_dict() -> dict[str, dict[str, list[int]]]:
    """Create a modalities to channel groups dict fixture for testing."""
    return {
        "s2": {"rgb": [0, 1, 2], "nir": [3]},
        "latlon": {"pos": [0, 1]},
    }
