"""Create windows with random timestamps."""

import argparse
import random
from datetime import datetime, timedelta, timezone

import shapely
from rslearn.const import WGS84_PROJECTION
from rslearn.dataset.window import Window
from rslearn.utils.geometry import STGeometry
from rslearn.utils.get_utm_ups_crs import get_utm_ups_projection
from upath import UPath

# Some arbitrarily chosen locations for now.
LOCATIONS = [
    (-122.32, 47.62),
    (-122.68, 45.52),
    (-121.49, 38.58),
    (-122.42, 37.78),
    (-96.80, 32.78),
    (-84.39, 33.76),
    (-80.84, 35.23),
    (-71.06, 42.36),
    (2.35, 48.85),
    (51.53, 25.29),
    (103.99, 1.36),
    (104.92, 11.57),
    (135.49, 34.71),
    (111.90, 43.72),
    (106.55, 29.56),
    (111.74, 27.26),
    (78.88, 17.51),
]

# Resolutions to sample from in m/pixel.
RESOLUTIONS = [1, 10, 250]

WINDOW_SIZE = 256

START_TIME = datetime(2016, 6, 1, tzinfo=timezone.utc)
END_TIME = datetime(2024, 6, 1, tzinfo=timezone.utc)
WINDOW_DURATION = timedelta(days=14)
GROUP = "default_{resolution}"


def create_window_random_time(
    ds_path: UPath, lon: float, lat: float, resolution: float
) -> Window:
    """Create a window centered at the specified longitude and latitude.

    It will have a random timestamp between START_TIME and END_TIME.

    Args:
        ds_path: path to the rslearn dataset to add the window to.
        lon: the longitude center.
        lat: the latitude center.
        resolution: the m/pixel resolution of the window. Note that it will be in UTM
            projection.

    Returns:
        the new Window.
    """
    projection = get_utm_ups_projection(lon, lat, resolution, -resolution)
    src_geom = STGeometry(WGS84_PROJECTION, shapely.Point(lon, lat), None)
    dst_geom = src_geom.to_projection(projection)
    bounds = (
        int(dst_geom.shp.x) - WINDOW_SIZE // 2,
        int(dst_geom.shp.y) - WINDOW_SIZE // 2,
        int(dst_geom.shp.x) + WINDOW_SIZE // 2,
        int(dst_geom.shp.y) + WINDOW_SIZE // 2,
    )
    total_seconds = (END_TIME - START_TIME).total_seconds()
    selected_seconds = random.randint(0, int(total_seconds))
    selected_ts = START_TIME + timedelta(seconds=selected_seconds)
    selected_date = datetime(
        selected_ts.year, selected_ts.month, selected_ts.day, tzinfo=timezone.utc
    )
    time_range = (selected_date, selected_date + WINDOW_DURATION)

    window_name = f"{str(projection.crs)}_{resolution}_{bounds[0]}_{bounds[1]}_{time_range[0].isoformat()}"
    group = GROUP.format(resolution=resolution)
    window = Window(
        path=Window.get_window_root(ds_path, group, window_name),
        group=group,
        name=window_name,
        projection=projection,
        bounds=bounds,
        time_range=time_range,
    )
    window.save()
    return window


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create windows with random timestamp for data ingestion",
    )
    parser.add_argument(
        "--ds_path",
        type=str,
        help="Dataset path",
        required=True,
    )
    args = parser.parse_args()

    ds_path = UPath(args.ds_path)

    for lon_base, lat_base in LOCATIONS:
        for lon_offset in [-0.03, 0, 0.03]:
            for lat_offset in [-0.03, 0, 0.03]:
                lon = lon_base + lon_offset
                lat = lat_base + lat_offset
                resolution = random.choice(RESOLUTIONS)
                create_window_random_time(ds_path, lon, lat, resolution)
