"""Utility functions for the data module."""

from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame

from helios.data.schemas import (
    Sentinel2FrequencyMetadataDataModel,
    Sentinel2MonthlyMetadataDataModel,
    TrainingDataIndexDataModel,
)

T = TypeVar("T")


def load_metadata(path: Path | str, schema: type[T], **kwargs: Any) -> DataFrame[T]:
    """Load metadata from a file with the specified schema.

    Args:
        path: Path to the file
        schema: Pandera schema class to validate against
        **kwargs: Additional arguments passed to pd.read_csv
    """
    # check the extension of the file say not implemented if not csv
    file_extension = (
        path.split(".")[-1] if isinstance(path, str) else path.name.split(".")[-1]
    )
    if file_extension not in ["csv", "parquet"]:
        raise NotImplementedError(f"File extension {file_extension} not supported")
    if file_extension == "csv":
        return pd.read_csv(path, **kwargs)
    elif file_extension == "parquet":
        return pd.read_parquet(path, **kwargs)
    else:
        raise NotImplementedError(f"File extension {file_extension} not supported")


@pa.check_types
def load_data_index(
    data_index_path: Path | str, **kwargs: Any
) -> DataFrame[TrainingDataIndexDataModel]:
    """Load the data index from a csv file."""
    return load_metadata(data_index_path, TrainingDataIndexDataModel, **kwargs)


@pa.check_types
def load_sentinel2_monthly_metadata(
    sentinel2_monthly_metadata_path: Path | str, **kwargs: Any
) -> DataFrame[Sentinel2MonthlyMetadataDataModel]:
    """Load the Sentinel-2 Monthly metadata from a csv file."""
    return load_metadata(
        sentinel2_monthly_metadata_path, Sentinel2MonthlyMetadataDataModel, **kwargs
    )


@pa.check_types
def load_sentinel2_frequency_metadata(
    sentinel2_frequency_metadata_path: Path | str, **kwargs: Any
) -> DataFrame[Sentinel2FrequencyMetadataDataModel]:
    """Load the Sentinel-2 Frequency metadata from a csv file."""
    return load_metadata(
        sentinel2_frequency_metadata_path, Sentinel2FrequencyMetadataDataModel, **kwargs
    )


# TODO: I just want a factory function that returns the metadata for a given data source and validates it
