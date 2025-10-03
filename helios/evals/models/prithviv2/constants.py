"""Constants for downloading dinov3 models from torch hub."""

from enum import StrEnum


class PrithviV2Models(StrEnum):
    """Names for different Prithvi models on torch hub."""

    VIT_300 = "Prithvi_EO_V2_300M"
    VIT_600 = "Prithvi_EO_V2_600M"


MODEL_TO_HF_INFO = {
    PrithviV2Models.VIT_300: {
        "hf_hub_id": f"ibm-nasa-geospatial/{PrithviV2Models.VIT_300.value}",
        "weights": f"{PrithviV2Models.VIT_300.value}.pt",
        "revision": "b2f2520ab889f42a25c5361ba18761fcb4ea44ad",
    },
    PrithviV2Models.VIT_600: {
        "hf_hub_id": f"ibm-nasa-geospatial/{PrithviV2Models.VIT_600.value}",
        "weights": f"{PrithviV2Models.VIT_600.value}.pt",
        "revision": "b2f2520ab889f42a25c5361ba18761fcb4ea44ad",
    },
}
