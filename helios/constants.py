"""Constants shared across the helios package."""

NAIP_BANDS = ["R", "G", "B", "IR"]

S2_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B10",
    "B11",
    "B12",
]

LATLON_BANDS = ["lat", "lon"]
TIMESTAMPS = ["day", "month", "year"]

WORLDCOVER_BANDS = ["B1"]

MODALITY_NAME_TO_BANDS = {
    "sentinel2": S2_BANDS,
    "latlon": LATLON_BANDS,
    "timestamps": TIMESTAMPS,
}


BASE_GSD = 10  # What unit is this in?
