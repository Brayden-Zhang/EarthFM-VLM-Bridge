"""Constants related to Helios dataset creation."""

MODALITIES = [
    "naip",
    "openstreetmap",
    "sentinel2_freq",
    "sentinel2_monthly",
    "worldcover",
]

# Columns in the per-modality metadata CSVs.
METADATA_COLUMNS = [
    "example_id",
    "image_idx",
    "start_time",
    "end_time",
]
