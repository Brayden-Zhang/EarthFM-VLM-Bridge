"""Input/Output handlers for different data source types."""

import json
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import rasterio
from class_registry import ClassRegistry
from einops import rearrange
from upath import UPath

from helios.constants import NAIP_BANDS, S2_BANDS, WORLDCOVER_BANDS


class DataSourceReader(ABC):
    """Base class for data source readers.

    All readers should return data in (H, W, T, C) format where:
    - H, W are spatial dimensions
    - T is number of timesteps (1 for static data)
    - C is number of channels/bands
    """

    @classmethod
    @abstractmethod
    def load(cls, file_path: UPath) -> Any:
        """Load data from file path.

        When Overiding this message defaults must be provided for extra kwargs
        """
        pass


DataSourceReaderRegistry = ClassRegistry[DataSourceReader]()


class TiffReader(DataSourceReader):
    """Base reader for Tiff files."""

    @classmethod
    def load(cls, file_path: UPath, bands: list[str] = []) -> np.ndarray:
        """Load data from a Tiff file."""
        if not bands:
            # Including a default to satisfy mypy
            raise ValueError("Bands must be provided")
        with rasterio.open(file_path) as data:
            values = data.read()
        num_timesteps = values.shape[0] / len(bands)
        if not num_timesteps.is_integer():
            raise ValueError(
                f"{file_path} has incorrect number of channels {bands} "
                f"{values.shape[0]=} {len(bands)=}"
            )
        num_timesteps = int(num_timesteps)

        data_array = rearrange(
            values, "(t c) h w -> h w t c", c=len(bands), t=num_timesteps
        )

        return data_array

    @classmethod
    def _check_bands(cls, bands: list[str], valid_bands: list[str]) -> None:
        """Check if the bands are valid."""
        if not all(band in valid_bands for band in bands):
            bands_not_in_valid = [band for band in bands if band not in valid_bands]
            raise ValueError(f"Invalid bands {bands_not_in_valid} for {cls.__name__}")


class GeoJSONReader(DataSourceReader):
    """Base reader for GeoJSON files."""

    @classmethod
    def load(cls, file_path: UPath) -> dict[str, Any]:
        """Load data from a GeoJSON file."""
        with file_path.open("r") as f:
            data = json.load(f)
        return data


@DataSourceReaderRegistry.register("sentinel2")
class Sentinel2Reader(TiffReader):
    """Reader for Sentinel-2 data."""

    @classmethod
    def load(cls, file_path: UPath, bands: list[str] = S2_BANDS) -> np.ndarray:
        """Load Sentinel-2 data with specific band handling.

        Returns:
            Array of shape (H, W, T, C) where C is len(S2_BANDS)
        """
        cls._check_bands(bands, S2_BANDS)

        values = super().load(file_path, bands=bands)

        return values


@DataSourceReaderRegistry.register("worldcover")
class WorldCoverReader(TiffReader):
    """Reader for WorldCover data."""

    @classmethod
    def load(cls, file_path: UPath, bands: list[str] = WORLDCOVER_BANDS) -> np.ndarray:
        """Load WorldCover data.

        Returns:
            Array of shape (H, W, 1, C) containing land cover classes
        """
        cls._check_bands(bands, WORLDCOVER_BANDS)

        values = super().load(file_path, bands=bands)
        num_timesteps = values.shape[2]
        if num_timesteps != 1:
            raise ValueError(
                f"WorldCover data must have 1 timestep, got {num_timesteps}"
            )

        return values


@DataSourceReaderRegistry.register("openstreetmap")
class OpenStreetMapReader(GeoJSONReader):
    """Reader for OpenStreetMap data."""

    @classmethod
    def load(cls, file_path: UPath) -> dict[str, Any]:
        """Load OpenStreetMap data."""
        return super().load(file_path)


@DataSourceReaderRegistry.register("naip")
class NAIPReader(TiffReader):
    """Reader for NAIP imagery."""

    @classmethod
    def load(cls, file_path: UPath, bands: list[str] = NAIP_BANDS) -> np.ndarray:
        """Load NAIP data.

        Returns:
            Array of shape (H, W, 1, C) containing NAIP bands
        """
        cls._check_bands(bands, NAIP_BANDS)

        values = super().load(file_path, bands=bands)
        num_timesteps = values.shape[2]
        if num_timesteps != 1:
            raise ValueError(f"NAIP data must have 1 timestep, got {num_timesteps}")

        return values


if __name__ == "__main__":
    # all these except the 1 MB s2 file are very small
    test_osm_file = UPath(
        "gs://ai2-helios/data/20250115-sample-dataset-helios/openstreetmap/EPSG:32610_10_52606_-503872_2018-09-20T00:00:00+00:00.geojson"
    )
    test_sentinel2_file = UPath(
        "gs://ai2-helios/data/20250115-sample-dataset-helios/sentinel2_freq/EPSG:32648_1_494421_-1282451_2016-12-09T00:00:00+00:00.tif"
    )
    test_worldcover_file = UPath(
        "gs://ai2-helios/data/20250115-sample-dataset-helios/worldcover/EPSG:32610_10_54981_-417951_2020-11-15T00:00:00+00:00.tif"
    )
    test_naip_file = UPath(
        "gs://ai2-helios/data/20250115-sample-dataset-helios/naip/EPSG:32610_1_423334_-5041313_2020-07-19T00:00:00+00:00.tif"
    )

    # Load and print info for each data source
    # s2_data, s2_timesteps = Sentinel2Reader.load(test_sentinel2_file)
    # print(f"Sentinel2 data shape: {s2_data.shape}, timesteps: {s2_timesteps}")

    # wc_data, wc_timesteps = WorldCoverReader.load(test_worldcover_file)
    # print(f"WorldCover data shape: {wc_data.shape}, timesteps: {wc_timesteps}")

    osm_data = OpenStreetMapReader.load(test_osm_file)
    print(f"OpenStreetMap data shape: {osm_data.keys()} {len(osm_data['features'])}")
    print(osm_data["type"])
    print(osm_data["crs"])

    # naip_data, naip_timesteps = NAIPReader.load(test_naip_file)
    # print(f"NAIP data shape: {naip_data.shape}, timesteps: {naip_timesteps}")
