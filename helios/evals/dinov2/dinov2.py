"""DINOv2 model wrapper."""

import logging
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from olmo_core.config import Config
from torch import nn
from torchvision import transforms

from helios.nn.flexihelios import PoolingType
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)
# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


HELIOS_SENTINEL2_RGB_BANDS = [3, 2, 1]
HELIOS_LANDSAT_RGB_BANDS = [4, 3, 2]


class DINOv2(nn.Module):
    """Wrapper for the dinov2 model that can ingest MaskedHeliosSample objects."""

    def __init__(
        self,
        torchhub_id: str = "dinov2_vitb14",
        patch_size: int = 14,
        use_cls_token: bool = False,
    ):
        """Initialize the dinov2 wrapper.

        Args:
            torchhub_id: The torch hub model ID for dinov2
            patch_size: Patch size for the vision transformer (default 14)
            use_cls_token: Whether to use the cls token (default False)
        """
        super().__init__()
        self.patch_size = patch_size
        self.use_cls_token = use_cls_token
        # Load the model
        self._load_model(torchhub_id)

    def _load_model(self, torchhub_id: str):
        """Load the dinov2 model from torch hub."""
        # Hack to get around https://discuss.pytorch.org/t/torch-hub-load-gives-httperror-rate-limit-exceeded/124769
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        for attempt in range(2):
            try:
                self.model = torch.hub.load(
                    "facebookresearch/dinov2",
                    torchhub_id,
                )
                break
            except Exception as e:
                logger.warning(f"Error loading  model: {e}. Retrying in 5 seconds...")
                time.sleep(5)
        else:
            raise RuntimeError(
                f"Failed to load dinov2 model {torchhub_id} after retrying."
            )

    def _process_modality_data(
        self,
        data: torch.Tensor,
        modality: str,
        apply_imagenet_normalization: bool = False,
    ) -> list[torch.Tensor]:
        """Process individual modality data.

        Args:
            data: Input tensor of shape [B, H, W, T, C]

        Returns:
            Processed tensor of shape [B, C*T, H, W]
        """
        # Rearrange from "b h w t c -> b (c t) h w" for DinoV2/dinov2 format
        t_dim = data.shape[3]

        # Get original dimensions
        original_height = data.shape[2]
        data_list = []
        for i in range(t_dim):
            data_i = rearrange(data[:, :, :, i, :], "b h w c -> b c h w")
            # Subset to RGB channels
            if modality == "sentinel2_l2a":
                data_i = data_i[:, HELIOS_SENTINEL2_RGB_BANDS, :, :]
            elif modality == "landsat":
                data_i = data_i[:, HELIOS_LANDSAT_RGB_BANDS, :, :]

            if original_height == 1:
                # For pixel time series resize to single patch
                data_i = F.interpolate(
                    data_i,
                    size=(self.patch_size, self.patch_size),
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                # Resize the image to the dinov2 pre-training input size
                image_size = 224
                data_i = F.interpolate(
                    data_i,
                    size=(image_size, image_size),
                    mode="bilinear",
                    align_corners=False,
                )
            if apply_imagenet_normalization:
                # normalize the data
                normalize_transform = make_normalize_transform()
                data_i = normalize_transform(data_i)
            data_list.append(data_i)
        return data_list

    def prepare_input(
        self,
        masked_helios_sample: MaskedHeliosSample,
        apply_imagenet_normalization: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Prepare input for the dinov2 model from MaskedHeliosSample.

        Args:
            masked_helios_sample: Input MaskedHeliosSample object

        Returns:
            Dictionary with 'imgs' and 'chn_ids' keys for dinov2 model
        """
        # Process each modality
        input_data_timesteps = {}
        for modality in masked_helios_sample.modalities:
            if modality not in ["sentinel2_l2a", "landsat"]:
                continue  # Skip non-rgb modalities

            data = getattr(masked_helios_sample, modality)

            if data is None:
                continue

            # Process the modality data
            processed_data = self._process_modality_data(
                data, modality, apply_imagenet_normalization
            )
            for i, data_i in enumerate(processed_data):
                # start the list if it doesn't exist
                if i not in input_data_timesteps:
                    input_data_timesteps[i] = []
                input_data_timesteps[i].append(data_i)
            batch_size = processed_data[0].shape[0]

        if not input_data_timesteps:
            raise ValueError("No valid modalities found for processing")
        per_timestep_inputs = []
        for i, input_data_i in input_data_timesteps.items():
            # Concatenate all modality data along channel dimension
            concatenated_imgs = torch.cat(input_data_i, dim=1)

            per_timestep_inputs.append(concatenated_imgs)
        return per_timestep_inputs

    # pooling type is on the timesteps only right now
    def forward(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
        apply_imagenet_normalization: bool = False,
    ) -> torch.Tensor:
        """Forward pass through dinov2 model for classification.

        Args:
            masked_helios_sample: Input MaskedHeliosSample object

        Returns:
            Model embeddings
        """
        # Prepare input
        per_timestep_inputs = self.prepare_input(
            masked_helios_sample, apply_imagenet_normalization
        )
        # potentially will need to add a flag for segmentation
        output_features = []
        for data in per_timestep_inputs:
            if self.use_cls_token:
                timestep_output = self.model(data)
            else:
                timestep_output = self.model.forward_features(data)[
                    "x_norm_patchtokens"
                ].mean(dim=1)
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features

    def forward_features(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
        apply_imagenet_normalization: bool = False,
    ) -> torch.Tensor:
        """Forward pass through dinov2 model for segmentation.

        Args:
            x_dict: Input dictionary with 'imgs' and 'chn_ids' keys

        Returns:
        """
        # supports multi-timestep input single timestep output
        per_timestep_dinov2_inputs = self.prepare_input(
            masked_helios_sample, apply_imagenet_normalization
        )
        output_features = []
        for dinov2_input in per_timestep_dinov2_inputs:
            timestep_output = self.model.forward_features(dinov2_input)[
                "x_norm_patchtokens"
            ]
            num_tokens = timestep_output.shape[1]
            height = int(math.sqrt(num_tokens))
            timestep_output = rearrange(
                timestep_output, "b (h w) d -> b h w d", h=height, w=height
            )
            output_features.append(timestep_output.unsqueeze(0))
        # stack in the timestep dimension and take the mean or maybe the max?
        if pooling == PoolingType.MEAN:
            output_features = torch.cat(output_features, dim=0).mean(dim=0)
        elif pooling == PoolingType.MAX:
            output_features = torch.max(torch.cat(output_features, dim=0), dim=0)[0]
        return output_features

    def __call__(
        self,
        masked_helios_sample: MaskedHeliosSample,
        pooling: PoolingType = PoolingType.MEAN,
        apply_imagenet_normalization: bool = False,
    ) -> torch.Tensor:
        """Make the wrapper callable."""
        return self.forward(masked_helios_sample, pooling, apply_imagenet_normalization)


@dataclass
class DINOv2Config(Config):
    """olmo_core style config for DINOv2Wrapper."""

    torchhub_id: str = "dinov2_vitb14"
    patch_size: int = 14
    use_cls_token: bool = False

    def build(self) -> DINOv2:
        """Build the DINOv2 from this config."""
        return DINOv2(
            torchhub_id=self.torchhub_id,
            patch_size=self.patch_size,
            use_cls_token=self.use_cls_token,
        )
