"""Create the index.csv file that lists all the examples and available modalities."""

import argparse
import csv

from upath import UPath

from .const import MODALITIES

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

    helios_path = UPath(args.helios_path)

    # List the available modalities for each example.
    modalities_by_example: dict[str, list[str]] = {}
    for modality in MODALITIES:
        for fname in (helios_path / modality).iterdir():
            example_id = fname.name.split(".")[0]
            if example_id not in modalities_by_example:
                modalities_by_example[example_id] = []
            modalities_by_example[example_id].append(modality)

    # Now we can write the CSV.
    with (helios_path / "index.csv").open("w") as f:
        column_names = [
            "example_id",
            "projection",
            "resolution",
            "start_column",
            "start_row",
            "time",
        ]
        for modality in MODALITIES:
            column_names.append(modality)

        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()

        for example_id, example_modalities in modalities_by_example.items():
            projection, resolution, start_column, start_row, time = example_id.split(
                "_"
            )
            csv_row = dict(
                example_id=example_id,
                projection=projection,
                resolution=resolution,
                start_column=start_column,
                start_row=start_row,
                time=time,
            )
            for modality in MODALITIES:
                if modality in example_modalities:
                    csv_row[modality] = "y"
                else:
                    csv_row[modality] = "n"
            writer.writerow(csv_row)
