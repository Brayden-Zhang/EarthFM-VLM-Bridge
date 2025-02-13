"""Constants shared across the helios package.

Warning: this is only developed for raster data currently.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from helios.constants import MODALITY_NAME_TO_BANDS

# The highest resolution that we are working at.
# Everything else is a factor (which is a power of 2) coarser than this resolution.
BASE_RESOLUTION = 0.625

# The default image tile size.
# Some images may be smaller if they are stored at a coarser resolution compared to the
# resolution that the grid is based on.
IMAGE_TILE_SIZE = 256


def get_resolution(resolution_factor: int) -> float | int:
    """Compute the resolution.

    If it is an integer, then we cast it to int so that it works with the raw Helios
    dataset, where some files are named based on the integer. We may want to change
    this in the future to avoid the extra code here.
    """
    resolution = BASE_RESOLUTION * resolution_factor
    if float(int(resolution)) == resolution:
        return int(resolution)
    return resolution


@dataclass(frozen=True)
class BandSet:
    """A group of bands that is stored at the same resolution.

    Many modalities only have one band set, but some have different bands at different
    resolutions.
    """

    # List of band names.
    bands: Sequence[str]

    # Resolution is BASE_RESOLUTION * resolution_factor.
    # If resolution == 0, this means the data
    # does not vary in space (e.g. latlons)
    resolution_factor: int

    def __hash__(self) -> int:
        """Hash this BandSet."""
        return hash((tuple(self.bands), self.resolution_factor))

    def get_resolution(self) -> float:
        """Compute the resolution."""
        return get_resolution(self.resolution_factor)


@dataclass(frozen=True)
class Modality:
    """Modality specification."""

    name: str
    tile_resolution_factor: int
    band_sets: Sequence[BandSet]
    is_multitemporal: bool

    def __hash__(self) -> int:
        """Hash this Modality."""
        return hash(self.name)

    def get_tile_resolution(self) -> float:
        """Compute the tile resolution."""
        return get_resolution(self.tile_resolution_factor)

    def bandsets_as_indices(self) -> list[list[int]]:
        """Return the band sets as indices."""
        modality_bands = MODALITY_NAME_TO_BANDS[self.name]

        band_specs_as_indices = []
        for band_set in self.band_sets:
            band_specs_as_indices.append(
                [modality_bands.index(b_name) for b_name in band_set.bands]
            )
        return band_specs_as_indices

    @property
    def num_band_sets(self) -> int:
        """Get the number of band sets."""
        return len(self.band_sets)

    @property
    def num_channels(self) -> int:
        """Get the number of channels.

        The number of channels is the sum of the number of bands in all the band sets.
        """
        return sum(len(band_set.bands) for band_set in self.band_sets)


MODALITIES = {
    "naip": Modality(
        name="naip",
        tile_resolution_factor=1,
        band_sets=[BandSet(["R", "G", "B", "IR"], 1)],
        is_multitemporal=False,
    ),
    "sentinel1": Modality(
        name="sentinel1",
        tile_resolution_factor=16,
        band_sets=[BandSet(["VV", "VH"], 16)],
        is_multitemporal=True,
    ),
    "sentinel2": Modality(
        name="sentinel2",
        tile_resolution_factor=16,
        band_sets=[
            # 10 m/pixel bands.
            BandSet(["B02", "B03", "B04", "B08"], 16),
            # 20 m/pixel bands.
            BandSet(["B05", "B06", "B07", "B8A", "B11", "B12"], 32),
            # 60 m/pixel bands that we store at 40 m/pixel.
            BandSet(["B01", "B09", "B10"], 64),
        ],
        is_multitemporal=True,
    ),
    "landsat": Modality(
        name="landsat",
        tile_resolution_factor=16,
        band_sets=[
            # 15 m/pixel bands that we store at 10 m/pixel.
            BandSet(["B8"], 16),
            # 30 m/pixel bands that we store at 20 m/pixel.
            BandSet(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11"], 16),
        ],
        is_multitemporal=True,
    ),
    "worldcover": Modality(
        name="worldcover",
        tile_resolution_factor=16,
        band_sets=[BandSet(["B1"], 16)],
        is_multitemporal=False,
    ),
    "openstreetmap": Modality(
        name="openstreetmap",
        tile_resolution_factor=16,
        band_sets=[BandSet(["aerialway_pylon", "aerodrome", "airstrip", "amenity_fuel", "building", "chimney", "communications_tower", "crane", "flagpole", "fountain", "generator_wind", "helipad", "highway", "leisure", "lighthouse", "obelisk", "observatory", "parking", "petroleum_well", "power_plant", "power_substation", "power_tower", "river", "runway", "satellite_dish", "silo", "storage_tank", "taxiway", "water_tower", "works"], 4)],
        is_multitemporal=False,
    ),
    "latlon": Modality(
        name="latlon",
        tile_resolution_factor=0,
        band_sets=[BandSet(["lat", "lon"], 0)],
        is_multitemporal=False,
    ),
}

# TODO: should latlon be a modality?
SUPPORTED_MODALITIES = ["sentinel1", "sentinel2", "worldcover", "latlon"]
TIMESTAMPS = ["day", "month", "year"]