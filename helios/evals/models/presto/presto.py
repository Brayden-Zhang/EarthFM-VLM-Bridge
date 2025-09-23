"""Presto wrapper to ingest Masked Helios Samples."""

import logging
from dataclasses import dataclass

import torch
from einops import repeat
from olmo_core.config import Config
from torch import nn
from upath import UPath

from helios.data.constants import Modality
from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample

from .single_file_presto import (
    NUM_DYNAMIC_WORLD_CLASSES,
    PRESTO_BANDS,
    PRESTO_S1_BANDS,
    PRESTO_S2_BANDS,
    Presto,
)

logger = logging.getLogger(__name__)

INPUT_PRESTO_BANDS = [b for b in PRESTO_BANDS if b != "B09"]
INPUT_PRESTO_S2_BANDS = [b for b in PRESTO_S2_BANDS if b != "B09"]


PRESTO_S1_SUBTRACT_VALUE = -25.0
PRESTO_S1_DIV_VALUE = 25.0
PRESTO_S2_SUBTRACT_VALUE = 0.0
PRESTO_S2_DIV_VALUE = 1e4


class PrestoWrapper(nn.Module):
    """Class containing the Presto model that can ingest MaskedHeliosSample objects."""

    supported_modalities = [Modality.SENTINEL2_L2A.name, Modality.SENTINEL1.name]
    requires_timeseries: bool = True

    def __init__(
        self,
        load_directory: str = "/weka/dfive-default/helios/models/presto",
        use_pretrained_normalizer: bool = True,
    ):
        """Initialize the Presto wrapper.

        Args:
            load_directory: The directory to load from
            use_pretrained_normalizer: Whether or not to apply presto pretraining normalization
        """
        super().__init__()

        self.use_pretrained_normalizer = use_pretrained_normalizer

        model = Presto.construct()
        model.load_state_dict(
            torch.load(UPath(load_directory) / "default_model.pt", map_location="cpu")
        )

        self.model = model.encoder
        self.kept_s2_band_idx = [
            i
            for i, v in enumerate(Modality.SENTINEL2_L2A.band_order)
            if v in INPUT_PRESTO_S2_BANDS
        ]
        self.kept_s1_band_idx = [
            i
            for i, v in enumerate(Modality.SENTINEL1.band_order)
            if v in PRESTO_S1_BANDS
        ]
        kept_s2_band_names = [
            val
            for val in Modality.SENTINEL2_L2A.band_order
            if val in INPUT_PRESTO_S2_BANDS
        ]
        kept_s1_band_names = [
            val for val in Modality.SENTINEL1.band_order if val in PRESTO_S1_BANDS
        ]
        self.to_presto_s2_map = [PRESTO_BANDS.index(val) for val in kept_s2_band_names]
        self.to_presto_s1_map = [PRESTO_BANDS.index(val) for val in kept_s1_band_names]

        self.month = 6  # default month

    def _preproccess(
        self,
        s2: torch.Tensor | None = None,
        s1: torch.Tensor | None = None,
        months: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # images should have shape (b h w c) or (b h w t c)
        if s2 is not None:
            data_device = s2.device
            if len(s2.shape) == 4:
                b, h, w, c_s2 = s2.shape
                t = 1
                s2 = repeat(torch.mean(s2, dim=(1, 2)), "b d -> b t d", t=1)
            else:
                assert len(s2.shape) == 5
                b, h, w, t, c_s2 = s2.shape
                s2 = torch.mean(s2, dim=(1, 2))

            x = torch.zeros(
                (b, t, len(INPUT_PRESTO_BANDS)), dtype=s2.dtype, device=s2.device
            )
            if self.use_pretrained_normalizer:
                s2 = (s2 - PRESTO_S2_SUBTRACT_VALUE) / PRESTO_S2_DIV_VALUE
            x[:, :, self.to_presto_s2_map] = s2[:, :, self.kept_s2_band_idx]

        elif s1 is not None:
            data_device = s1.device
            if len(s1.shape) == 4:
                b, h, w, c_s1 = s1.shape
                t = 1
                s1 = repeat(torch.mean(s1, dim=(1, 2)), "b d -> b t d", t=1)
            else:
                assert len(s1.shape) == 5
                b, h, w, t, c_s1 = s1.shape
                s1 = torch.mean(s1, dim=(1, 2))

            # add a single timestep
            x = torch.zeros(
                (b, t, len(INPUT_PRESTO_BANDS)), dtype=s1.dtype, device=s1.device
            )
            if self.use_pretrained_normalizer:
                s1 = (s1 - PRESTO_S1_SUBTRACT_VALUE) / PRESTO_S1_DIV_VALUE
            x[:, :, self.to_presto_s1_map] = s1[:, :, self.kept_s1_band_idx]

        else:
            raise ValueError("no s1 or s2?")
        s_t_m = torch.ones(
            (b, t, len(INPUT_PRESTO_BANDS)),
            dtype=x.dtype,
            device=x.device,
        )
        if s2 is not None:
            s_t_m[:, :, self.to_presto_s2_map] = 0
        elif s1 is not None:
            s_t_m[:, :, self.to_presto_s1_map] = 0

        if months is None:
            months = torch.ones((b, t), device=data_device) * self.month
        else:
            assert months.shape[-1] == t

        dymamic_world = (
            torch.ones((b, t), device=data_device) * NUM_DYNAMIC_WORLD_CLASSES
        )

        return (
            x,
            s_t_m,
            dymamic_world.long(),
            months.long(),
        )

    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
        spatial_pool: bool = False,
    ) -> torch.Tensor:
        """Forward pass through presto model."""
        if pooling == PoolingType.MAX:
            # should this throw an exception?
            logger.warning("MAX pooling ignored by presto")

        s2 = getattr(masked_helios_sample, Modality.SENTINEL2_L2A.name)
        s1 = getattr(masked_helios_sample, Modality.SENTINEL1.name)
        months = masked_helios_sample.timestamps[:, :, 1]

        x, mask, dynamic_world, months = self._preproccess(s2=s2, s1=s1, months=months)
        output = self.model(
            x=x, dynamic_world=dynamic_world, mask=mask, month=months, eval_task=True
        )  # [B, self.dim]
        if spatial_pool:
            return repeat(output, "b d -> b h w d", h=1, w=1)
        else:
            return output


@dataclass
class PrestoConfig(Config):
    """olmo_core style config for PrestoWrapper."""

    load_directory: str = "/weka/dfive-default/helios/models/presto"
    use_pretrained_normalizer: bool = True

    def build(self) -> PrestoWrapper:
        """Build the Presto model."""
        return PrestoWrapper(
            load_directory=self.load_directory,
            use_pretrained_normalizer=self.use_pretrained_normalizer,
        )
