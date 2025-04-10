"""Run the conversion of a dataset to h5py files."""
import logging
import sys

from helios.data.constants import Modality
from olmo_core.utils import prepare_cli_environment
from helios.dataset.convert_to_h5py import ConvertToH5pyConfig

logger = logging.getLogger(__name__)

def build_default_config() -> ConvertToH5pyConfig:
    """Build the default configuration for H5 conversion."""
    return ConvertToH5pyConfig(
        tile_path="",
        supported_modality_names=[
            Modality.SENTINEL2_L2A.name,
            Modality.SENTINEL1.name,
            Modality.WORLDCOVER.name,
            Modality.SRTM.name,
            Modality.NAIP.name,
            Modality.LANDSAT.name,
            Modality.OPENSTREETMAP_RASTER.name,
        ],
        multiprocessed_h5_creation=True,
    )


def main(config_builder=build_default_config, *args):
    """Parse arguments, build config, and run the H5 conversion."""
    prepare_cli_environment()

    script, *overrides = sys.argv

    # Create the configuration object from arguments
    default_config = config_builder()
    config = default_config.merge(overrides)
    logger.info(f"Configuration overrides: {overrides}")
    logger.info(f"Configuration loaded: {config}")

    # Build and run the converter
    converter = config.build()
    logger.info("Starting H5 conversion process...")
    converter.run()
    logger.info("H5 conversion process finished.")


if __name__ == "__main__":
    main()
