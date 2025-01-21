"""Utilities related to dataset creation."""

from datetime import datetime

from upath import UPath


class ExampleMetadata:
    """Class to represent the various metadata encoded within an example ID."""

    def __init__(
        self,
        projection: str,
        resolution: float,
        start_column: int,
        start_row: int,
        time: datetime,
    ):
        """Create a new ExampleMatadata.

        Args:
            projection: the UTM projection that the example is in.
            resolution: the m/pixel resolution of the example.
            start_column: starting column in pixel units.
            start_row: starting row in pixel units.
            time: the center time of the example.
        """
        self.projection = projection
        self.resolution = resolution
        self.start_column = start_column
        self.start_row = start_row
        self.time = time

    def get_example_id(self) -> str:
        """Encode the metadata back to an example ID."""
        return (
            f"{self.projection}_{self.resolution}_"
            + f"{self.start_column}_{self.start_row}_"
            + self.time.isoformat()
        )


def parse_example_id(example_id: str) -> ExampleMetadata:
    """Parse the specified example ID, extracting the encoded metadata.

    Args:
        example_id: the example ID to parse.

    Returns:
        ExampleMetadata object containing the metadata encoded within the example ID
    """
    projection, resolution, start_column, start_row, time = example_id.split("_")
    return ExampleMetadata(
        projection,
        float(resolution),
        int(start_column),
        int(start_row),
        datetime.fromisoformat(time),
    )


def get_modality_dir(helios_path: UPath, modality: str) -> UPath:
    """Get the directory where data should be stored for the specified modality.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality.

    Returns:
        directory within helios_path to store the modality.
    """
    return helios_path / modality


def list_examples_for_modality(helios_path: UPath, modality: str) -> list[str]:
    """List the example IDs available for the specified modality.

    This is determined by listing the contents of the modality directory. Index and
    metadata CSVs are not used.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality to check.

    Returns:
        a list of example IDs
    """
    modality_dir = get_modality_dir(helios_path, modality)
    if not modality_dir.exists():
        return []

    # We just list the directory and strip the extension.
    example_ids = []
    for fname in modality_dir.iterdir():
        example_ids.append(fname.name.split(".")[0])
    return example_ids


def get_modality_fname(
    helios_path: UPath, modality: str, example_id: str, ext: str
) -> UPath:
    """Get the filename where to store data for the specified example and modality.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality name.
        example_id: the example ID.
        ext: the filename extension, like "tif" or "geojson".

    Returns:
        the filename to store the data in.
    """
    return helios_path / modality / f"{example_id}.{ext}"


def get_modality_temp_meta_dir(helios_path: UPath, modality: str) -> UPath:
    """Get the directory to store per-example metadata files for a given modality.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality name.

    Returns:
        the directory to store the metadata files.
    """
    return helios_path / f"{modality}_meta"


def get_modality_temp_meta_fname(
    helios_path: UPath, modality: str, example_id: str
) -> UPath:
    """Get the temporary filename to store the metadata for an example and modality.

    This is created by the helios.dataset_creation.rslearn_to_helios scripts. It will
    then be read by helios.dataset_creation.make_meta_summary to create the final
    metadata CSV.

    Args:
        helios_path: the Helios dataset root.
        modality: the modality name.
        example_id: the example ID.

    Returns:
        the filename for the per-example metadata CSV.
    """
    return get_modality_temp_meta_dir(helios_path, modality) / f"{example_id}.csv"
