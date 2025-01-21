"""Concatenate the metadata files for one modality."""

import argparse
import csv

from upath import UPath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create concatenated metadata file",
    )
    parser.add_argument(
        "--helios_path",
        type=str,
        help="Helios dataset path",
        required=True,
    )
    parser.add_argument(
        "--modality",
        type=str,
        help="Modality to summarize",
        required=True,
    )
    args = parser.parse_args()

    helios_path = UPath(args.helios_path)

    # Concatenate the CSVs while keeping the header only from the first file that we
    # read.
    column_names: list[str] | None = None
    csv_rows = []
    meta_dir = helios_path / f"{args.modality}_meta"
    for fname in meta_dir.iterdir():
        with fname.open() as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV at {fname} does not contain header")
            if column_names is None:
                column_names = list(reader.fieldnames)
            for csv_row in reader:
                csv_rows.append(csv_row)

    if column_names is None:
        raise ValueError(f"did not find any files in {meta_dir}")

    with (helios_path / f"{args.modality}.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        writer.writeheader()
        writer.writerows(csv_rows)
