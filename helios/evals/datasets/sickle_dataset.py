"""SICKLE dataset class."""

import os
import glob
import torch
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from pathlib import Path
from datetime import date
import albumentations as A
import cv2


MONTH_TO_INT = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
S1_BANDS = ['VV', 'VH']
# NOTE: We need to handle missing bands in L8
L8_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7", "ST_B10"]


class SICKLEProcessor:
    """Process SICKLE dataset into PyTorch objects."""

    def __init__(self, csv_path: Path, data_dir: Path, output_dir: Path):
        """Initialize SICKLE processor.

        Args:
            csv_path: Path to the CSV file.
            data_dir: Path to the data directory.
            output_dir: Path to the output directory.
        """
        self.csv_path = csv_path
        self.data_dir = UPath(data_dir)
        self.output_dir = UPath(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resize = A.Resize(height=32, width=32)

    def impute_l8_bands(self, img: torch.Tensor) -> torch.Tensor:
        """Impute missing bands in L8 images."""
        img = torch.stack(
            [
                img[0, ...],  # fill B1 with B1
                img[1, ...],  # fill B2 with B2
                img[2, ...],  # fill B3 with B3
                img[3, ...],  # fill B4 with B4
                img[4, ...],  # fill B5 with B5
                img[5, ...],  # fill B6 with B6
                img[6, ...],  # fill B7 with B7
                img[6, ...],  # fill B8 with B7, IMPUTED!
                img[6, ...],  # fill B9 with B7, IMPUTED!
                img[7, ...],  # fill B10 with B10
                img[7, ...],  # fill B11 with B10, IMPUTED!
            ]
        )

    def _read_mask(self, mask_path: Path) -> np.ndarray:
        """Read a mask from a path."""
        with rasterio.open(mask_path) as fp:
            mask = fp.read()
        
        # There're multiple layers in the mask, we only use the first two layers
        # Which is the plot_mask and crop_type_mask
        mask = mask[:2, ...]
        mask[0][mask[0] == 0] = -1
        # Convert crop type mask into binary (Paddy: 0, Non-Paddy: 1)
        # Reference: https://github.com/Depanshu-Sani/SICKLE/blob/main/utils/dataset.py
        mask[1] -= 1
        mask[1][mask[1] >= 1] = 1
        # Convert to -1 to ignore
        mask[mask < 0] = -1
        return mask

    def _get_image_date(self, image_path: Path) -> str:
        """Get the date of the image.
        
        Args:
            image_path: Path to the image.

        Returns:
            year-month string.
        """
        if "S2" in image_path.name:
            # For S2 2018 data?
            if os.path.basename(image_path)[0] == "T":
                image_date = os.path.basename(image_path).split("_")[1][:8]
            else:
                image_date = os.path.basename(image_path).split("_")[0][:8]
        elif "S1" in image_path.name:
            image_date = os.path.basename(image_path).split("_")[4][:8]
        elif "L8" in image_path.name:
            image_date = os.path.basename(image_path).split("_")[2][:8]
        
        return f"{image_date[:4]}-{image_date[4:6]}"  # year-month string

    def _aggregate_months(self, images: list[str], start_date: date, end_date: date) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate images by month.
        
        Args:
            images: List of image paths.
            start_date: Start date.
            end_date: End date.

        Returns:
            Tuple of aggregated images and dates.
        """
        images = sorted(images)
        # Get all the unique year-month strings between start_date and end_date
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            all_dates.append(current_date.strftime("%Y-%m"))
            current_date = current_date.replace(day=1) + timedelta(days=32)
            current_date = current_date.replace(day=1)
        all_dates = list(set(all_dates))
        all_dates.sort()

        dates_dict = dict[str, list[torch.Tensor]]()
        for date in all_dates:
            dates_dict[date] = []
        
        for image_path in images:
            image_date = self._get_image_date(image_path)
            dates_dict[image_date].append(image_path)
        
        img_list: list[torch.Tensor] = []
        date_list: list[str] = []
        for date in all_dates:
            if dates_dict[date]:
                stacked_imgs = torch.stack(dates_dict[date])
                month_avg = stacked_imgs.mean(dim=0)
                if len(img_list) < 12:
                    img_list.append(month_avg)
                    date_list.append(date)
        
        return torch.stack(img_list), torch.tensor(date_list, dtype=torch.long)

    def process_sample(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Process a single sample from the SICKLE dataset."""
        uid = sample["uid"]
        plot_id = sample["plot_id"]
        standard_season = sample["standard_season"]
        year = sample["year"]
        split = sample["split"]

        start_date = date(int(year), MONTH_TO_INT[standard_season.split("-")[0]], 1)
        end_date = date(int(year), MONTH_TO_INT[standard_season.split("-")[1]], 1)
        # Deal with the case where it across the year boundary
        if end_date <= start_date:
            end_date = end_date.replace(year=end_date.year + 1)

        # Get all the S2, S1, and L8 images for the sample
        s2_path = self.data_dir / f"images/S2/npy/{uid}/*.npz"
        s1_path = self.data_dir / f"images/S1/npy/{uid}/*.npz"
        l8_path = self.data_dir / f"images/L8/npy/{uid}/*.npz"
        s2_images = glob.glob(s2_path)
        s1_images = glob.glob(s1_path)
        l8_images = glob.glob(l8_path)
        
        # For now, we only use the 10m mask, there're also 3m, 30m masks
        mask_path = self.data_dir / f"masks/10m/{uid}.tif"
        mask = self._read_mask(mask_path)
        plot_mask, crop_type_mask = mask[0], mask[1]

        # Remove plots that are not in this split
        unmatched_plot_ids = set(np.unique(plot_mask)) - set(self.split_plot_ids[split])
        for unmatched_plot_id in unmatched_plot_ids:
            crop_type_mask[plot_mask == unmatched_plot_id] = -1
        
        # Resize the mask to 32*32
        crop_type_mask = crop_type_mask.transpose(1, 2, 0)
        crop_type_mask = self.resize(image=crop_type_mask)["image"].transpose(2, 0, 1)
        targets = torch.tensor(crop_type_mask, dtype=torch.long)

        return {
            "split": split,
            "s2_images": s2_images,
            "s1_images": s1_images,
            "l8_images": l8_images,
            "targets": targets,
        }

    
    def process(self) -> None:
        """Process the SICKLE dataset."""
        
        all_samples = []
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            sample = {
                "uid": row["UNIQUE_ID"],
                # NOTE: One sample may have multiple plot_ids
                "plot_id": row["PLOT_ID"],
                "standard_season": row["STANDARD_SEASON"],
                "year": row["YEAR"],
                "split": row["SPLIT"]
            }
            all_samples.append(sample)

        # Get the unique plot_ids per split
        self.split_plot_ids = defaultdict(list)
        for split in ["train", "val", "test"]:
            plot_ids = df[df["SPLIT"] == split]["PLOT_ID"].unique()
            self.split_plot_ids[split] = plot_ids
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.process_sample, all_samples))
        
        


