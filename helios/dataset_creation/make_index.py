"""Create the index.csv file that lists all the examples and available modalities."""

import argparse
import csv

from upath import UPath

from .const import INDEX_COLUMNS, MODALITIES
from .util import list_examples_for_modality, parse_example_id


def make_index(helios_path: UPath) -> None:
    """Create index.csv summary for this dataset.

    Data for each of the modalities should already be populated in the per-modality
    directories. This function lists the files in those directories, extracts the
    example IDs, and then writes the summary which includes which modalities are
    available for which example IDs.
    """
    # List the available modalities for each example.
    modalities_by_example: dict[str, list[str]] = {}
    for modality in MODALITIES:
        for example_id in list_examples_for_modality(helios_path, modality):
            if example_id not in modalities_by_example:
                modalities_by_example[example_id] = []
            modalities_by_example[example_id].append(modality)

    # Now we can write the CSV.
    with (helios_path / "index.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()

        for example_id, example_modalities in modalities_by_example.items():
            example_metadata = parse_example_id(example_id)
            csv_row = dict(
                example_id=example_id,
                projection=example_metadata.projection,
                resolution=str(example_metadata.resolution),
                start_column=int(example_metadata.start_column),
                start_row=int(example_metadata.start_row),
                time=example_metadata.time.isoformat(),
            )
            for modality in MODALITIES:
                if modality in example_modalities:
                    csv_row[modality] = "y"
                else:
                    csv_row[modality] = "n"
            writer.writerow(csv_row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create index.csv summary",
    )
    parser.add_argument(
        "--helios_path",
        type=str,
        help="Helios dataset path",
        required=True,
    )
    args = parser.parse_args()

    make_index(UPath(args.helios_path))
