"""Dataset module for helios."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import pandas as pd
import rioxarray
import torch
import xarray as xr
from einops import rearrange
from torch.utils.data import Dataset as PyTorchDataset


class DatasetOutput(NamedTuple):
    """A named tuple for storing the output of a dataset.

    Args:
        space_time_x: The space-time input data.
        space_x: The space input data.
        time_x: The time input data.
        static_x: The static input data.
        months: The months input data.
    """

    space_time_x: np.ndarray
    space_x: np.ndarray
    time_x: np.ndarray
    static_x: np.ndarray
    months: np.ndarray

    @classmethod
    def concatenate(cls, datasetoutputs: Sequence["DatasetOutput"]) -> "DatasetOutput":
        """Concatenate a sequence of DatasetOutput objects.

        Args:
            datasetoutputs: A sequence of DatasetOutput objects to concatenate.

        Returns:
            A new DatasetOutput object with the concatenated data.
        """
        s_t_x = np.stack([o.space_time_x for o in datasetoutputs], axis=0)
        sp_x = np.stack([o.space_x for o in datasetoutputs], axis=0)
        t_x = np.stack([o.time_x for o in datasetoutputs], axis=0)
        st_x = np.stack([o.static_x for o in datasetoutputs], axis=0)
        months = np.stack([o.months for o in datasetoutputs], axis=0)
        return cls(s_t_x, sp_x, t_x, st_x, months)


def load_data_index(data_index_path: Path) -> pd.DataFrame:
    """Load the data index from a csv file."""
    columns = ["path", "start_date", "end_date", "location", "other metadata"]
    # How do we want to store and load this index? perhaps switch to parquet?
    return pd.read_csv(data_index_path)


def to_cartesian(
    lat: float | np.ndarray | torch.Tensor, lon: float | np.ndarray | torch.Tensor
) -> np.ndarray | torch.Tensor:
    """Convert latitude and longitude to Cartesian coordinates

    Args:
        lat: The latitude.
        lon: The longitude.

    Returns:
        The Cartesian coordinates.
    """
    if isinstance(lat, float):
        assert (
            -90 <= lat <= 90
        ), f"lat out of range ({lat}). Make sure you are in EPSG:4326"
        assert (
            -180 <= lon <= 180
        ), f"lon out of range ({lon}). Make sure you are in EPSG:4326"
        assert isinstance(lon, float), f"Expected float got {type(lon)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return np.array([x, y, z])
    elif isinstance(lon, np.ndarray):
        assert (
            -90 <= lat.min()
        ), f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        assert (
            90 >= lat.max()
        ), f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        assert (
            -180 <= lon.min()
        ), f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        assert (
            180 >= lon.max()
        ), f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        assert isinstance(lat, np.ndarray), f"Expected np.ndarray got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_np = np.cos(lat) * np.cos(lon)
        y_np = np.cos(lat) * np.sin(lon)
        z_np = np.sin(lat)
        return np.stack([x_np, y_np, z_np], axis=-1)
    elif isinstance(lon, torch.Tensor):
        assert (
            -90 <= lat.min()
        ), f"lat out of range ({lat.min()}). Make sure you are in EPSG:4326"
        assert (
            90 >= lat.max()
        ), f"lat out of range ({lat.max()}). Make sure you are in EPSG:4326"
        assert (
            -180 <= lon.min()
        ), f"lon out of range ({lon.min()}). Make sure you are in EPSG:4326"
        assert (
            180 >= lon.max()
        ), f"lon out of range ({lon.max()}). Make sure you are in EPSG:4326"
        assert isinstance(lat, torch.Tensor), f"Expected torch.Tensor got {type(lat)}"
        # transform to radians
        lat = lat * math.pi / 180
        lon = lon * math.pi / 180
        x_t = torch.cos(lat) * torch.cos(lon)
        y_t = torch.cos(lat) * torch.sin(lon)
        z_t = torch.sin(lat)
        return torch.stack([x_t, y_t, z_t], dim=-1)
    else:
        raise AssertionError(f"Unexpected input type {type(lon)}")


# Perhaps a preprocessor module that does transforms on the loaded tif and or splits it out into the multiple arrays for model input


# TODO: Adding a Dataset specific fingerprint is probably good for an evolving dataset
class HeliosDataset(PyTorchDataset):
    """Helios dataset."""

    def __init__(self, data_index_path: Path, output_hw: int = 256):
        self.data_index_path = data_index_path
        self.data_index = load_data_index(data_index_path)
        self.output_hw = output_hw
        # TODO: Appropriate Interface for file access

    def __len__(self) -> int:
        return len(self.data_index)

    def _tif_to_array(self, tif_path: Path) -> DatasetOutput:
        """Convert a tif file to an array.

        Args:
            tif_path: The path to the tif file.

        Returns:
            A DatasetOutput object with the data from the tif file.
        """
        with cast(xr.Dataset, rioxarray.open_rasterio(tif_path)) as data:
            # [all_combined_bands, H, W]
            # all_combined_bands includes all dynamic-in-time bands
            # interleaved for all timesteps
            # followed by the static-in-time bands
            values = cast(np.ndarray, data.values)
            lon = np.mean(cast(np.ndarray, data.x)).item()
            lat = np.mean(cast(np.ndarray, data.y)).item()

        # this is a bit hackey but is a unique edge case for locations,
        # which are not part of the exported bands but are instead
        # computed here
        static_bands_in_tif = len(EO_STATIC_BANDS) - len(LOCATION_BANDS)

        num_timesteps = (
            values.shape[0] - len(SPACE_BANDS) - static_bands_in_tif
        ) / len(ALL_DYNAMIC_IN_TIME_BANDS)
        assert num_timesteps % 1 == 0, f"{tif_path} has incorrect number of channels"
        dynamic_in_time_x = rearrange(
            values[: -(len(SPACE_BANDS) + static_bands_in_tif)],
            "(t c) h w -> h w t c",
            c=len(ALL_DYNAMIC_IN_TIME_BANDS),
            t=int(num_timesteps),
        )
        dynamic_in_time_x = self._fillna(dynamic_in_time_x, EO_DYNAMIC_IN_TIME_BANDS_NP)
        space_time_x = dynamic_in_time_x[:, :, :, : -len(TIME_BANDS)]

        # calculate indices, which have shape [h, w, t, 1]
        ndvi = self.calculate_ndi(space_time_x, band_1="B8", band_2="B4")

        space_time_x = np.concatenate((space_time_x, ndvi), axis=-1)

        time_x = dynamic_in_time_x[:, :, :, -len(TIME_BANDS) :]
        time_x = np.nanmean(time_x, axis=(0, 1))

        space_x = rearrange(
            values[-(len(SPACE_BANDS) + static_bands_in_tif) : -static_bands_in_tif],
            "c h w -> h w c",
        )
        space_x = self._fillna(space_x, np.array(SPACE_BANDS))

        static_x = values[-static_bands_in_tif:]
        # add DW_STATIC and WC_STATIC
        dw_bands = space_x[
            :, :, [i for i, v in enumerate(SPACE_BANDS) if v in DW_BANDS]
        ]
        wc_bands = space_x[
            :, :, [i for i, v in enumerate(SPACE_BANDS) if v in WC_BANDS]
        ]
        static_x = np.concatenate(
            [
                np.nanmean(static_x, axis=(1, 2)),
                to_cartesian(lat, lon),
                np.nanmean(dw_bands, axis=(0, 1)),
                np.nanmean(wc_bands, axis=(0, 1)),
            ]
        )
        static_x = self._fillna(static_x, np.array(STATIC_BANDS))

        months = self.month_array_from_file(tif_path, int(num_timesteps))

        try:
            assert not np.isnan(space_time_x).any(), f"NaNs in s_t_x for {tif_path}"
            assert not np.isnan(space_x).any(), f"NaNs in sp_x for {tif_path}"
            assert not np.isnan(time_x).any(), f"NaNs in t_x for {tif_path}"
            assert not np.isnan(static_x).any(), f"NaNs in st_x for {tif_path}"
            assert not np.isinf(space_time_x).any(), f"Infs in s_t_x for {tif_path}"
            assert not np.isinf(space_x).any(), f"Infs in sp_x for {tif_path}"
            assert not np.isinf(time_x).any(), f"Infs in t_x for {tif_path}"
            assert not np.isinf(static_x).any(), f"Infs in st_x for {tif_path}"
            return DatasetOutput(
                space_time_x.astype(np.half),
                space_x.astype(np.half),
                time_x.astype(np.half),
                static_x.astype(np.half),
                months,
            )
        except AssertionError as e:
            raise e

    def _tif_to_array_with_checks(self, idx):
        tif_path = self.tifs[idx]
        try:
            output = self._tif_to_array(tif_path)
            return output
        except Exception as e:
            # TODO: Not sure we want to keep this we are essentially increasing prob of another example if we find a bad example that fails
            # should this be possible after rslearn has ingested already?
            print(f"Replacing tif {tif_path} due to {e}")
            if idx == 0:
                new_idx = idx + 1
            else:
                new_idx = idx - 1
            self.tifs[idx] = self.tifs[new_idx]
            tif_path = self.tifs[idx]
        output = self._tif_to_array(tif_path)
        return output

    def __getitem__(self, index: int) -> DatasetOutput:
        return self.data_index.iloc[index]
