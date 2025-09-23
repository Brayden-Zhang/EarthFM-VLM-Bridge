"""AnySat wrapper to ingest MaskedHeliosSample."""

import logging
from dataclasses import dataclass
from datetime import datetime

import torch
from einops import rearrange, repeat
from olmo_core.config import Config
from torch import nn

from helios.data.constants import Modality
from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)


class AnySat(nn.Module):
    """AnySat wrapper for MaskedHelioSample."""

    # these are the bands which AnySat accepts
    # https://github.com/gastruc/AnySat?tab=readme-ov-file#format-your-data
    ANYSAT_S2_BAND_ORDERING = [
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    ]
    ANYSAT_S1_BAND_ORDERING = ["vv", "vh", "ratio"]
    ANYSAT_L8_BAND_ORDERING = [
        "B8",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B9",
        "B10",
        "B11",
    ]

    helios_modalities_to_anysat_names = {
        Modality.SENTINEL2_L2A.name: "s2",
        Modality.LANDSAT.name: "l8",
        Modality.SENTINEL1.name: "s1",
    }

    supported_modalities = [
        Modality.SENTINEL2_L2A.name,
        Modality.LANDSAT.name,
        Modality.SENTINEL1.name,
    ]

    def __init__(self) -> None:
        """AnySat wrapper."""
        super().__init__()
        self.model = torch.hub.load(
            "gastruc/anysat",
            "anysat",
            pretrained=True,
            flash_attn=True,
            force_reload=True,
        )

        self.modality_to_band_indices = {
            Modality.SENTINEL2_L2A.name: [
                Modality.SENTINEL2_L2A.band_order.index(v)
                for v in self.ANYSAT_S2_BAND_ORDERING
            ],
            Modality.SENTINEL1.name: [
                Modality.SENTINEL1.band_order.index(v)
                for v in self.ANYSAT_S1_BAND_ORDERING
                if v in Modality.SENTINEL1.band_order
            ],
            Modality.LANDSAT.name: [
                Modality.LANDSAT.band_order.index(v)
                for v in self.ANYSAT_L8_BAND_ORDERING
                if v in Modality.LANDSAT.band_order
            ],
        }
        self.month = 5  # default month, if none is given (indexing from 0)

    @staticmethod
    def _month_day_to_day_of_year(
        months: torch.Tensor, days: torch.Tensor | None = None
    ) -> torch.Tensor:
        output_tensors = []
        for i in range(months.shape[0]):
            output_tensors.append(
                torch.tensor(
                    [datetime(2025, m + 1, 1).timetuple().tm_yday for m in months[i]]
                ).to(device=months.device)
            )
        months_as_doy = torch.stack(output_tensors)
        if days is not None:
            months_as_doy += days - 1
        return months_as_doy

    def _calculate_patch_size(self, h: int) -> int:
        # based on https://arxiv.org/pdf/2412.14123, a patch size of
        # 40 is the minimum used for images of 128x128. Since smaller patches
        # = more tokens, this should lead to the best performance
        h_in_m = h * 10
        patch_size = min(40, h_in_m)
        return patch_size

    def _process_modality_data(self, data: torch.Tensor, modality: str) -> torch.Tensor:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, H, W, T, C]
            modality: What modality data is

        Returns:
            tensor of shape [B, T, C, H, W]
        """
        # no model specific normalization - the authors recommend
        # "standard dataset normalization"
        data = rearrange(data, "b h w t c -> b t c h w")
        data = data[:, :, self.modality_to_band_indices[modality], :, :]
        if modality == Modality.SENTINEL1.name:
            # add the ratio
            ratio_band = data[:, :, :1, :, :] / (data[:, :, 1:, :, :] + 1e-6)
            data = torch.concat((data, ratio_band), dim=2)
        return data

    def prepare_input(
        self,
        masked_helios_sample: MaskedHeliosSample,
    ) -> dict[str, torch.Tensor]:
        """Prepare input for the AnySat model from MaskedHeliosSample."""
        input_data: dict[str, dict[str, torch.Tensor]] = {}

        for modality in masked_helios_sample.modalities:
            if modality not in self.helios_modalities_to_anysat_names.keys():
                logger.warning(
                    f"Skipping modality {modality} as it is not in the supported "
                    f"modalities list {self.helios_modalities_to_anysat_names.keys()}"
                )
                continue

            data = getattr(masked_helios_sample, modality)

            if data is None:
                continue

            processed_data = self._process_modality_data(data, modality)
            # Process the modality data
            input_data[self.helios_modalities_to_anysat_names[modality]] = (
                processed_data
            )
            num_timesteps = processed_data.shape[1]
            if num_timesteps > 1:
                assert masked_helios_sample.timestamps is not None

            # Note that time series requires a _dates companion tensor containing the day of
            # the year: 01/01 = 0, 31/12=364.
            if masked_helios_sample.timestamps is None:
                months = repeat(
                    torch.tensor([self.month]).to(device=processed_data.device),
                    "d -> b d",
                    b=processed_data.shape[0],
                )
                doy = self._month_day_to_day_of_year(months=months)
            else:
                months = masked_helios_sample.timestamps[:, :, 1]
                days = masked_helios_sample.timestamps[:, :, -1]
                doy = self._month_day_to_day_of_year(months=months, days=days)
            input_data[f"{self.helios_modalities_to_anysat_names[modality]}_dates"] = (
                doy
            )
        return input_data

    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the AnySat model."""
        processed_inputs = self.prepare_input(masked_helios_sample)
        if pooling == PoolingType.MAX:
            raise ValueError("Unsupported pooling type MAX for AnySat.")

        hs = []
        for key, val in processed_inputs.items():
            if not key.endswith("dates"):
                hs.append(val.shape[-1])
        if len(set(hs)) != 1:
            raise RuntimeError("Expected all inputs to have the same dimension")
        patch_size = self._calculate_patch_size(hs[0])

        # from the README:
        # "The sub patches are 1x1 pixels for time series and 10x10 pixels for VHR images.
        # If using output='dense', specify the output_modality."
        # Let's preferentially use output_modality in this order: [s2, s1, landsat]
        input_modalities = [
            k for k in processed_inputs.keys() if not k.endswith("dates")
        ]
        if "s2" in input_modalities:
            output_modality = "s2"
        elif "s1" in input_modalities:
            output_modality = "s1"
        elif "l8" in input_modalities:
            output_modality = "l8"
        else:
            raise RuntimeError(
                f"Expected one of s2, s1, l8 in input modalities, got {input_modalities}"
            )

        if spatial_pool:
            output = "dense"
        else:
            output = "tile"
        output_patches = self.model(
            x=processed_inputs,
            patch_size=patch_size,
            output=output,
            output_modality=output_modality,
        )
        return output_patches


@dataclass
class AnySatConfig(Config):
    """olmo_core style config for AnySat."""

    def build(self) -> AnySat:
        """Build the Croma model."""
        return AnySat()
