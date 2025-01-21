"""Test make_index script."""

import csv
import pathlib

from upath import UPath

from helios.dataset_creation.const import MODALITIES
from helios.dataset_creation.make_index import make_index
from helios.dataset_creation.util import get_modality_fname, parse_example_id

# An example ID to use for testing.
TEST_EXAMPLE_ID = "EPSG:32610_10_123_456_2024-01-01T00:00:00+00:00"


def test_example_id(tmp_path: pathlib.Path) -> None:
    """Ensure make_index populates the example metadata columns correctly."""
    # Create a single example with a single modality.
    helios_path = UPath(tmp_path)
    modality = MODALITIES[0]
    fname = get_modality_fname(helios_path, modality, TEST_EXAMPLE_ID, "tif")
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.touch()
    make_index(helios_path)

    with (helios_path / "index.csv").open() as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    assert len(csv_rows) == 1
    csv_row = csv_rows[0]
    assert csv_row["projection"] == "EPSG:32610"
    assert csv_row["resolution"] == "10.0"
    assert csv_row["start_column"] == "123"
    assert csv_row["start_row"] == "456"
    assert csv_row["time"] == "2024-01-01T00:00:00+00:00"


def test_one_example_all_modalities(tmp_path: pathlib.Path) -> None:
    """Test make_index when we have a single example with every modality present."""
    helios_path = UPath(tmp_path)
    for modality in MODALITIES:
        fname = get_modality_fname(helios_path, modality, TEST_EXAMPLE_ID, "tif")
        fname.parent.mkdir(parents=True, exist_ok=True)
        fname.touch()
    make_index(helios_path)

    with (helios_path / "index.csv").open() as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    assert len(csv_rows) == 1
    csv_row = csv_rows[0]
    for modality in MODALITIES:
        assert csv_row[modality] == "y"


def test_one_example_per_modality(tmp_path: pathlib.Path) -> None:
    """Test make_index when we have one example per modality.

    Here, each example contains data for a single modality.
    """
    helios_path = UPath(tmp_path)

    # We increment the start row for each modality.
    example_id_tmpl = "EPSG:32610_10_{start_column}_456_2024-01-01T00:00:00+00:00"
    for modality_idx, modality in enumerate(MODALITIES):
        example_id = example_id_tmpl.format(start_column=modality_idx)
        fname = get_modality_fname(helios_path, modality, example_id, "tif")
        fname.parent.mkdir(parents=True, exist_ok=True)
        fname.touch()
    make_index(helios_path)

    with (helios_path / "index.csv").open() as f:
        reader = csv.DictReader(f)
        csv_rows = list(reader)

    assert len(csv_rows) == len(MODALITIES)
    for csv_row in csv_rows:
        example_id = csv_row["example_id"]
        example_metadata = parse_example_id(example_id)

        for modality_idx, modality in enumerate(MODALITIES):
            if modality_idx == example_metadata.start_column:
                assert csv_row[modality] == "y"
            else:
                assert csv_row[modality] == "n"
