"""PASTIS dataset class."""

# # Dataset is between September 2018 to November 2019

# # load this torch object: /weka/dfive-default/presto_eval_sets/pastis/pastis_train.pt
# # print out the shape of the object inside

# import torch

# pastis_train = torch.load("/weka/dfive-default/presto_eval_sets/pastis/pastis_train.pt")

# unique_months = set()

# for item in pastis_train:
#     print(item, pastis_train[item].shape)
#     if item == "months":
#         for item2 in pastis_train[item]:
#             unique_months.add("_".join(str(item2)))

# print(unique_months)

# # for item in pastis_train:
# #     if item == "images":
# #         for item2 in pastis_train[item]:
# #             # print min and max of the tensor at dimension 1
# #             print(item2.min(), item2.max())
# # range -1 to 18

# target_set = set()

# for item in pastis_train:
#     if item == "targets":
#         for item2 in pastis_train[item]:
#             target_set.add(int(item2.max()))

# print(target_set)  #

# # tensor(-132.6000) tensor(14567.)

# # # images torch.Size([5820, 12, 13, 64, 64])
# # # months torch.Size([5820, 12])
# # # targets torch.Size([5820, 64, 64])

# # # already subset to 64 * 64 images

# # # tensor([ 9, 10, 11, 12,  1,  2,  3,  4,  5,  6,  7,  8])
# # # tensor([ 9, 10, 11, 12,  1,  2,  3,  4,  5,  6,  7,  8])

import json
from pathlib import Path

import einops
import torch
import torch.multiprocessing
from torch.utils.data import Dataset
from upath import UPath

from helios.data.constants import Modality
from helios.data.dataset import HeliosSample
from helios.train.masking import MaskedHeliosSample

torch.multiprocessing.set_sharing_strategy("file_system")

PASTIS_DIR = UPath("/weka/dfive-default/presto_eval_sets/pastis")


class PASTISDataset(Dataset):
    """PASTIS dataset class."""

    def __init__(
        self,
        path_to_splits: Path,
        split: str,
        partition: str,
        norm_stats_from_pretrained: bool = True,
        norm_method: str = "norm_no_clip",
    ):
        """Init PASTIS dataset.

        Args:
            path_to_splits: Path where .pt objects returned by process_mados have been saved
            split: Split to use
            partition: Partition to use
            norm_stats_from_pretrained: Whether to use normalization stats from pretrained model
            norm_method: Normalization method to use, only when norm_stats_from_pretrained is False
        """
        assert split in ["train", "val", "valid", "test"]
        if split == "valid":
            split = "val"

        self.split = split
        self.norm_method = norm_method

        self.norm_stats_from_pretrained = norm_stats_from_pretrained
        # If normalize with pretrained stats, we initialize the normalizer here
        if self.norm_stats_from_pretrained:
            from helios.data.normalize import Normalizer, Strategy

            self.normalizer_computed = Normalizer(Strategy.COMPUTED)

        torch_obj = torch.load(path_to_splits / f"pastis_{split}.pt")
        self.images = torch_obj["images"]
        self.labels = torch_obj["targets"]
        self.months = torch_obj["months"]

        if (partition != "default") and (split == "train"):
            with open(path_to_splits / f"{partition}_partition.json") as json_file:
                subset_indices = json.load(json_file)

            self.images = self.images[subset_indices]
            self.labels = self.labels[subset_indices]
            self.months = self.months[subset_indices]

    def __len__(self) -> int:
        """Length of the dataset."""
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[MaskedHeliosSample, torch.Tensor]:
        """Return a single PASTIS data instance."""
        image = self.images[idx]  # (12, 13, 64, 64)
        labels = self.labels[idx]  # (64, 64)
        months = self.months[idx]  # (12)

        image = einops.rearrange(image, "t c h w -> h w t c")
        if self.norm_stats_from_pretrained:
            image = self.normalizer_computed.normalize(Modality.SENTINEL2_L2A, image)

        timestamps = []
        for month in months:
            if month != 1:
                year = 2018
            else:
                year = 2019
            timestamps.append(torch.tensor([1, month, year], dtype=torch.long))
        timestamps = torch.stack(timestamps)

        masked_sample = MaskedHeliosSample.from_heliossample(
            HeliosSample(
                sentinel2_l2a=torch.tensor(image).float(), timestamps=timestamps
            )
        )

        return masked_sample, labels
