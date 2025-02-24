"""Unit tests for the Helios Dataset Visualization."""

import pytest

from helios.data.dataset import HeliosDataset
from helios.data.visualize import visualize_sample





def test_visualize_sample():
    """Test the visualize_sample function."""
    dataset = HeliosDataset()
    visualize_sample(dataset, 0)
