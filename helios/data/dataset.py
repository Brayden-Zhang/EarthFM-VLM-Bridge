"""Dataset module for helios."""

import hashlib
import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from einops import rearrange
from olmo_core.aliases import PathOrStr
from olmo_core.distributed.utils import get_fs_local_rank
from pyproj import Transformer
from torch.utils.data import Dataset
from upath import UPath

from helios.constants import LATLON_BANDS, TIMESTAMPS
from helios.data.constants import (BASE_RESOLUTION, IMAGE_TILE_SIZE,
                                   MODALITIES, SUPPORTED_MODALITIES, Modality)
from helios.dataset.parse import TimeSpan
from helios.dataset.sample import SampleInformation, load_image_for_sample
from helios.types import ArrayTensor

logger = logging.getLogger(__name__)


class HeliosSample(NamedTuple):
    """A sample of the data from the Helios dataset.

    This is a namedtuple that contains the data of a single sample or a batch of samples from the Helios dataset.
    For each modality, we have an ArrayTensor named by the modality, along with the latlon and timestamps.
    """

    sentinel2: ArrayTensor | None = None  # [B, H, W, T, len(S2_bands)]
    sentinel1: ArrayTensor | None = None  # [B, H, W, T, len(S1_bands)]
    worldcover: ArrayTensor | None = None  # [B, H, W, len(WC_bands)]
    latlon: ArrayTensor | None = None  # [B, 2]
    timestamps: ArrayTensor | None = None  # [B, T, D=3], where D=[day, month, year]

    def shape(self, attribute: str, mask: bool = False) -> Sequence[int]:
        """Returns the expected shape of an attribute.

        This is useful if you want to know what the shape of a
        missing attribute would have been for this sample.

        Args:
            attribute: The attribute to get the shape of, e.g., "sentinel2", "timestamps", etc.
            mask: Whether to get the shape of the mask.

        Returns:
            The shape of the attribute.
        """
        # It is safe to assume we always have Sentinel2, timestamps, and latlon
        # If other attributes are missing, we use Sentinel2 to get its partial shape (B, H, W, T)
        # For static modality like worldcover, we specify the T dimension as 1
        if attribute == "timestamps":
            if not mask:
                return self.timestamps.shape
            else:
                # timestamps is a special case which is not in Modality
                raise ValueError("Timestamps are not maskable")
        else:
            if self.sentinel2 is None:
                raise ValueError("Sentinel2 is not present in the sample")
            return_shape = []
            if MODALITIES.get(attribute).get_tile_resolution() > 0:
                return_shape += self.sentinel2.shape[
                    :-2
                ]  # get batch size (if has), height, width
            if MODALITIES.get(attribute).is_multitemporal:
                return_shape += [self.sentinel2.shape[-2]]  # number of timesteps
            if not mask:
                # TODO: change it to num_bands instead
                return_shape += [
                    len(MODALITIES.get(attribute).get_band_names())
                ]  # number of bands
            else:
                return_shape += [len(MODALITIES.get(attribute).band_sets)]
            return return_shape

    @staticmethod
    def num_bands(attribute: str) -> int:
        """Get the number of bands for a given attribute."""
        if attribute == "timestamps":
            return len(TIMESTAMPS)
        else:
            return len(MODALITIES.get(attribute).get_band_names())

    def as_dict(self, ignore_nones: bool = True) -> dict[str, ArrayTensor | None]:
        """Convert the namedtuple to a dictionary.

        Args:
            ignore_nones: Whether to ignore None values.

        Returns:
            Dictionary representation of the namedtuple.
        """
        return_dict = {}
        for field in self._fields:
            val = getattr(self, field)
            if ignore_nones and (val is None):
                continue
            else:
                return_dict[field] = val
        return return_dict

    def to_device(self, device: torch.device) -> "HeliosSample":
        """Move all tensors to the specified device.

        Args:
            device: The device to move the tensors to.

        Returns:
            A new HeliosSample with all tensors moved to the specified device.
        """
        return HeliosSample(
            sentinel2=self.sentinel2.to(device) if self.sentinel2 is not None else None,
            sentinel1=self.sentinel1.to(device) if self.sentinel1 is not None else None,
            worldcover=(
                self.worldcover.to(device) if self.worldcover is not None else None
            ),
            latlon=self.latlon.to(device) if self.latlon is not None else None,
            timestamps=(
                self.timestamps.to(device) if self.timestamps is not None else None
            ),
        )


def collate_helios(batch: list[HeliosSample]) -> HeliosSample:
    """Collate function."""

    # Stack tensors while handling None values
    def stack_or_none(attr: str) -> torch.Tensor | None:
        """Stack the tensors while handling None values."""
        if batch[0].__getattribute__(attr) is None:
            return None
        return torch.stack(
            [torch.from_numpy(getattr(sample, attr)) for sample in batch], dim=0
        )

    return HeliosSample(
        sentinel2=stack_or_none("sentinel2"),
        sentinel1=stack_or_none("sentinel1"),
        worldcover=stack_or_none("worldcover"),
        latlon=stack_or_none("latlon"),
        timestamps=stack_or_none("timestamps"),
    )


class HeliosDataset(Dataset):
    """Helios dataset."""

    def __init__(
        self,
        *samples: SampleInformation,
        path: UPath,
        dtype: np.dtype = np.float32,
    ):
        """Initialize the dataset.

        Warning from OLMo-core:
            In distributed settings, be sure that the :data:`work_dir` is shared among all local ranks
            and :data:`fs_local_rank` is set accordingly. Once those fields are set you should then call
            :meth:`prepare()` in the main process before doing anything else.

        Args:
            samples: The samples to include in the dataset.
            path: The path to the dataset.
            dtype: The dtype of the data.
        """
        self.samples = self._filter_samples(list(samples))
        self.path = path
        self.dtype = dtype
        self._fs_local_rank = get_fs_local_rank()
        self._work_dir: Path | None = None  # type: ignore
        self._work_dir_set = False

    @property
    def fingerprint_version(self) -> str:
        """The version of the fingerprint."""
        return "v0.1"

    @property
    def fingerprint(self) -> str:
        """Can be used to identify/compare a dataset."""
        sha256_hash = hashlib.sha256()
        sha256_hash.update(
            f"path={self.path},"
            f"sample_size={len(self.samples)},"
            f"dtype={self.dtype}".encode()
        )
        return sha256_hash.hexdigest()

    @property
    def fs_local_rank(self) -> int:
        """Get the fs local rank."""
        return self._fs_local_rank

    @fs_local_rank.setter
    def fs_local_rank(self, _fs_local_rank: int) -> None:
        """Set the fs local rank."""
        self._fs_local_rank = _fs_local_rank

    @property
    def work_dir(self) -> Path:
        """Get the working directory."""
        if self._work_dir is not None:
            return self._work_dir
        else:
            return Path(tempfile.gettempdir())

    @work_dir.setter
    def work_dir(self, _work_dir: PathOrStr) -> None:
        """Set the working directory."""
        self._work_dir = Path(_work_dir)
        self._work_dir_set = True

    @property
    def work_dir_set(self) -> bool:
        """Check if the working directory was explicitly set."""
        return self._work_dir_set

    def prepare(self) -> None:
        """Prepare the dataset."""
        len(self)

    def _filter_samples(
        self, samples: list[SampleInformation]
    ) -> list[SampleInformation]:
        """Filter samples to adjust to the HeliosSample format."""
        # Right now, only use yearly S1 and S2 data
        logger.info(f"Number of samples before filtering: {len(samples)}")
        filtered_samples = []
        for sample in samples:
            # Check if the sample has all the necessary modalities
            if not all(
                modality in sample.modalities for modality in SUPPORTED_MODALITIES
            ):
                continue
            # Check if S1 and S2 all have 12 months of data
            total_months = 12
            if (
                sample.time_span != TimeSpan.YEAR
                or len(sample.modalities[MODALITIES.get("sentinel1")].images)
                != total_months
                or len(sample.modalities[MODALITIES.get("sentinel2")].images)
                != total_months
            ):
                logger.info(
                    f"Sample {sample.grid_tile} has {len(sample.modalities[MODALITIES.get('sentinel1')].images)} months of S1 data and {len(sample.modalities[MODALITIES.get('sentinel2')].images)} months of S2 data"
                )
                continue
            filtered_samples.append(sample)
        logger.info(f"Number of samples after filtering: {len(filtered_samples)}")
        return filtered_samples

    def _get_latlon(self, sample: SampleInformation) -> np.ndarray:
        """Get the latlon of the sample."""
        # Get coordinates at projection units, and then transform to latlon
        grid_resolution = sample.grid_tile.resolution_factor * BASE_RESOLUTION
        x, y = (
            (sample.grid_tile.col + 0.5) * grid_resolution * IMAGE_TILE_SIZE,
            (sample.grid_tile.row + 0.5) * -grid_resolution * IMAGE_TILE_SIZE,
        )
        transformer = Transformer.from_crs(
            sample.grid_tile.crs, "EPSG:4326", always_xy=True
        )
        lon, lat = transformer.transform(x, y)
        return np.array([lat, lon])

    def _get_timestamps(self, sample: SampleInformation) -> np.ndarray:
        """Get the timestamps of the sample."""
        sample_sentinel2 = sample.modalities[MODALITIES.get("sentinel2")]
        timestamps = [i.start_time for i in sample_sentinel2.images]
        dt = pd.to_datetime(timestamps)
        # Note that month should be 0-indexed
        return np.array([dt.day, dt.month - 1, dt.year]).T

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> HeliosSample:
        """Get the item at the given index."""
        sample = self.samples[index]
        sample_dict = {}
        for modality in sample.modalities:
            # Skip modalities that are not supported right now
            if modality not in SUPPORTED_MODALITIES:
                continue
            sample_modality = sample.modalities[modality]
            image = load_image_for_sample(sample_modality, sample)
            modality_data = rearrange(image, "t c h w -> h w t c")
            sample_dict[modality.name] = modality_data
            # Get latlon and timestamps from s2
            if modality == MODALITIES.get("sentinel2"):
                sample_dict["latlon"] = self._get_latlon(sample)
                sample_dict["timestamps"] = self._get_timestamps(sample)
        # TODO: Add normalization and better way of doing dtype
        return HeliosSample(**sample_dict)
