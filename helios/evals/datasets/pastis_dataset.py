"""PASTIS dataset class."""

# Dataset is between September 2018 to November 2019

# load this torch object: /weka/dfive-default/presto_eval_sets/pastis/pastis_train.pt
# print out the shape of the object inside

import torch

pastis_train = torch.load("/weka/dfive-default/presto_eval_sets/pastis/pastis_train.pt")

unique_months = set()

for item in pastis_train:
    print(item, pastis_train[item].shape)
    if item == "months":
        for item2 in pastis_train[item]:
            unique_months.add("_".join(str(item2)))

print(unique_months)

# for item in pastis_train:
#     if item == "images":
#         for item2 in pastis_train[item]:
#             # print min and max of the tensor at dimension 1
#             print(item2.min(), item2.max())
# range -1 to 18


for item in pastis_train:
    if item == "targets":
        for item2 in pastis_train[item]:
            print(item2.min(), item2.max())

# tensor(-132.6000) tensor(14567.)

# # images torch.Size([5820, 12, 13, 64, 64])
# # months torch.Size([5820, 12])
# # targets torch.Size([5820, 64, 64])

# # already subset to 64 * 64 images

# # tensor([ 9, 10, 11, 12,  1,  2,  3,  4,  5,  6,  7,  8])
# # tensor([ 9, 10, 11, 12,  1,  2,  3,  4,  5,  6,  7,  8])

# import json
# import os
# from pathlib import Path

# import numpy as np
# import torch
# import torch.multiprocessing
# import torch.nn.functional as F
# from einops import repeat
# from PIL import Image
# from torch.utils.data import Dataset
# from upath import UPath

# from helios.data.constants import Modality
# from helios.data.dataset import HeliosSample
# from helios.train.masking import MaskedHeliosSample

# from .constants import EVAL_S2_BAND_NAMES, EVAL_TO_HELIOS_S2_BANDS
# from .normalize import normalize_bands

# torch.multiprocessing.set_sharing_strategy("file_system")

# PASTIS_DIR = UPath("/weka/dfive-default/presto_eval_sets/pastis")


# class PASTISDataset(Dataset):
#     """PASTIS dataset class."""

#     default_day_month_year = [1, 9, 2018]

#     def __init__(
#         self,
#         path_to_splits: Path,
#         split: str,
#         partition: str,
#         norm_stats_from_pretrained: bool = False,
#     )
