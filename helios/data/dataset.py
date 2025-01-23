"""Dataset module for helios."""

import logging
from typing import Any

import numpy as np
from olmo_core.data.numpy_dataset import NumpyDatasetBase
from torch.utils.data import Dataset
from upath import UPath

from helios.data.data_source_io import DataSourceReader, DataSourceReaderRegistry

logger = logging.getLogger(__name__)


class HeliosDataset(NumpyDatasetBase, Dataset):
    """Helios dataset."""

    def __init__(self, *samples: dict[str, Any], dtype: np.dtype):
        """Initialize the dataset.

        Things that would need to be optional or should be forgotten about, or changed
        - paths would need to ba dictionary or collection of paths for this to work
        - pad_token_id: int,
        - eos_token_id: int,
        - vocab_size: int,
        """
        super().__init__(
            *samples,
            dtype=dtype,
            pad_token_id=-1,  # Not needed only LM
            eos_token_id=-1,  # Not needed only LM
            vocab_size=-1,  # Not needed only LM
        )

    @property
    def max_sequence_length(self) -> int:
        """Max sequence length."""
        # NOT SUPER needed
        return max(item["num_timesteps"] for item in self.paths)

    @property
    def fingerprint(self) -> str:
        """Fingerprint of the dataset."""
        # LM specific
        raise NotImplementedError("Fingerprint not implemented")

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.paths)

    def _load_data_source(
        self, file_path: UPath | str, data_source: str
    ) -> np.ndarray | dict[str, Any]:
        """Load data from a data source using the appropriate reader.

        Args:
            file_path: Path to the data file
            data_source: Name of the data source

        Returns:
            Either a numpy array or a dictionary of data
        """
        try:
            reader: DataSourceReader = DataSourceReaderRegistry.get_class(data_source)
            return reader.load(file_path)
        except Exception as e:
            logger.error(f"Error loading {file_path} from {data_source}: {e}")
            raise

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get the item at the given index."""
        sample = self.paths[index]
        data_source_paths = sample["data_source_paths"]
        data_inputs = {}

        for data_source, file_path in data_source_paths.items():
            data_input = self._load_data_source(file_path, data_source)
            data_inputs[data_source] = data_input
        return {
            "data_inputs": data_inputs,
            "sample_metadata": sample["sample_metadata"],
            "data_source_metadata": sample["data_source_metadata"],
        }
