"""Trying to prototype fitting everything into olmo core."""

import logging

import numpy as np
import torch
from einops import rearrange
from olmo_core.utils import setup_logging
from upath import UPath

from helios.data.collator import per_modality_collate_fn
from helios.data.dataloader import HeliosDataLoader
from helios.data.dataset import HeliosDataset
from helios.dataset.index import DatasetIndexParser
from helios.train.trainer import HeliosTrainer

logger = logging.getLogger(__name__)


## Config does not yet support our new dataset type so we will construct manually for now

if __name__ == "__main__":
    setup_logging()
    # set log level to debug
    logger.setLevel(logging.DEBUG)

    index_path_old = "gs://ai2-helios/data/20250113-sample-dataset-helios/index.csv"
    index_path = "gs://ai2-helios/data/20250115-sample-dataset-helios/index.csv"
    index_parser = DatasetIndexParser(index_path)
    samples = index_parser.samples
    workdir = UPath("/Users/henryh/Desktop/eai-repos/helios-repos/helios/workdir")
    dataloader = HeliosDataLoader.wrap_numpy_dataset(
        dataset=HeliosDataset(
            *samples,
            ignore_data_sources=["openstreetmap"],
            filter_samples_with_missing_inputs=True,
            dtype=np.dtype("float32"),
        ),
        global_batch_size=4,
        dp_world_size=1,
        collator=per_modality_collate_fn,
        work_dir=workdir,
        num_threads=0,
    )
    model = torch.hub.load(
        "gastruc/anysat",
        "anysat",
        pretrained=False,
        force_reload=True,
        flash_attn=False,
        release=False,
        scales={"all": [1, 2, 4, 8]},
        num_patches={"all": 16},
        modalities={"all": ["s2", "naip"]},
    )
    from copy import deepcopy

    from helios.train.mock_jepa import JEPAAny
    from helios.train.predictor import MockPredictor

    predictor = MockPredictor(
        num_patches=100,
        embed_dim=768,
        predictor_embed_dim=16,
        depth=1,
        num_heads=1,
        mlp_ratio=1.0,
        qkv_bias=True,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        flash_attn=False,
    )
    model.model.modalities = {"all": ["s2", "naip"]}
    jepa = JEPAAny(
        encoder=model.model,
        predictor=predictor,
        ratio=0.5,
    )
    target_encoder = deepcopy(jepa.encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # # potentially missing dataset prepare
    for epoch in range(1, 3):
        dataloader.reshuffle(epoch=epoch)
        batch_iterator = dataloader._iter_batches()
        # Need to call reshuffle
        batches_found = 0
        for batch in batch_iterator:
            with torch.no_grad():
                input_batch = {}
                input_batch["dataset"] = "all"
                input_batch["label"] = torch.tensor([])
                input_batch["scale"] = 1
                input_batch["s2"] = rearrange(
                    batch.sentinel2, "b h w t c -> b t c h w"
                )[:, :, :10, :, :]
                input_batch["s2_dates"] = batch.sentinel2_time_indices
                input_batch["naip"] = rearrange(batch.naip, "b h w t c -> b t c h w")[
                    :, :, :10, :, :
                ]
                input_batch["naip_dates"] = batch.naip_time_indices

                # We want a dataset string data source and dates and so on, and scale label as well
                # wants our datasource to be called name name is all the modalities
                # wants a dict of stacked per modality uses s2 instead of sentinel2
                # also wants a dataes function that takes a batch and returns a dict of stacked per modality
                h = target_encoder.forward(input_batch)
                # h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                # B = len(h)
                # h = apply_masks(h, mask_pred)
                # h = repeat_interleave_batch(h, B, repeat=len(mask_enc))
            print(batch)
            break
        dataloader.reset()

    # Need an optimizer
    # Need a checkpointer
    # Need a module
    # first lets grab the anysat model already in there repo

    # from olmo_core.optim import AdamWConfig
    # from olmo_core.train.checkpoint import CheckpointerConfig
    # from olmo_core.train.common import Duration

    # max_duration = Duration.epochs(4)

    # checkpointer_config = CheckpointerConfig(work_dir=workdir)
    # checkpointer = checkpointer_config.build()
    # DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # optim_config = AdamWConfig()
    # optim = optim_config.build(model)
    # trainer = HeliosTrainer(
    #     work_dir=workdir,
    #     model=model,
    #     optim=optim,
    #     data_loader=dataloader,
    #     device=DEVICE,
    #     save_folder=workdir / "save_folder",
    #     callbacks={},
    #     rank_microbatch_size=4,
    #     max_duration=max_duration,
    #     checkpointer=checkpointer,
    # )
    # trainer.fit()
