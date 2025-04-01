"""Finetune the Helios model on a downstream task."""

import json
import logging
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from olmo_core.distributed.checkpoint import (
    load_model_and_optim_state,
)
from torch.utils.data import DataLoader

from helios.evals.datasets import get_eval_dataset
from helios.evals.datasets.configs import EvalDatasetConfig, TaskType
from helios.evals.datasets.utils import eval_collate_fn
from helios.evals.utils import adjust_learning_rate
from helios.nn.flexihelios import Encoder, EncoderConfig, PoolingType, PredictorConfig
from helios.nn.latent_mim import LatentMIMConfig
from helios.train.masking import MaskedHeliosSample

logger = logging.getLogger(__name__)

FT_LRs = [1e-5, 3e-5, 6e-5, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 6e-3]


class PredictionHead(nn.Module):
    """Prediction head for a downstream task."""

    def __init__(self, encoder: Encoder, task_type: TaskType, num_classes: int) -> None:
        """Initialize the prediction head."""
        super().__init__()

        self.encoder = deepcopy(encoder)
        if task_type == TaskType.CLASSIFICATION:
            self.head = nn.Linear(encoder.embedding_size, num_classes)
        else:
            raise ValueError(f"Invalid task type: {task_type}")

    def forward(self, batch: MaskedHeliosSample) -> torch.Tensor:
        """Forward pass."""
        batch_features = self.encoder(batch, patch_size=4)
        batch_embeddings = batch_features.pool_unmasked_tokens(
            PoolingType.MEAN, spatial_pooling=False
        )
        output = self.head(batch_embeddings)
        return output


def load_config(checkpoint_input_dir: Path) -> dict:
    """Load the config file from the checkpoint input directory."""
    assert (
        checkpoint_input_dir / "config.json"
    ).exists(), f"Config file not found at {checkpoint_input_dir}"

    with open(checkpoint_input_dir / "config.json", encoding="utf-8") as f:
        config_dict = json.load(f)

    return config_dict


def train_and_eval_finetune(
    config: EvalDatasetConfig,
    lr: float,
    data_loader: DataLoader,
    device: torch.device,
    grid_size: int,
    batch_size: int,
    epochs: int = 50,
) -> None:
    """Run a linear probe on the Helios model."""
    finetune_cls(
        data_loader=data_loader,
        lr=lr,
        epochs=epochs,
        num_classes=config.num_classes,
        device=device,
        checkpoint_input_dir=Path(
            "/weka/dfive-default/helios/checkpoints/henryh/3latentmim_tiny_masking_modality_loss_patch_discrimination_new_token_exit_zero/step154400"
        ),
    )


def finetune_cls(
    data_loader: DataLoader,
    checkpoint_input_dir: Path,
    num_classes: int,
    # patch_size: int = 4,
    # pooling_type: PoolingType = PoolingType.MEAN,
    lr: float = 1e-4,
    epochs: int = 50,
    device: torch.device = torch.device("cuda"),
) -> nn.Module:
    """Finetune the Helios model on a downstream task."""
    config = load_config(checkpoint_input_dir)
    model_config = config["model"]
    # skip any key that is not in the LatentMIMConfig
    encoder_config = model_config["encoder_config"]
    del encoder_config["_CLASS_"]
    decoder_config = model_config["decoder_config"]
    del decoder_config["_CLASS_"]
    transform_type = model_config["transform_type"]

    model_config = LatentMIMConfig(
        encoder_config=EncoderConfig(**encoder_config),
        decoder_config=PredictorConfig(**decoder_config),
        transform_type=transform_type,
    )

    model = model_config.build()

    load_model_and_optim_state(checkpoint_input_dir / "model_and_optim", model)
    encoder = model.encoder

    finetuned_model = PredictionHead(
        encoder=encoder, task_type=TaskType.CLASSIFICATION, num_classes=10
    ).to(device)

    finetuned_model = finetuned_model.train()
    opt = torch.optim.AdamW(finetuned_model.parameters(), lr=lr)

    loss_function = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch} of {epochs}")
        for i, batch in enumerate(data_loader):
            masked_helios_sample, label = batch
            label = label.to(device=device)

            # Convert modality data into bfloat16
            masked_helios_sample_dict = masked_helios_sample.as_dict(return_none=False)
            for key, val in masked_helios_sample_dict.items():
                if key == "timestamps":
                    masked_helios_sample_dict[key] = val.to(device=device)
                else:
                    masked_helios_sample_dict[key] = val.to(
                        device=device, dtype=torch.bfloat16
                    )

            masked_helios_sample = MaskedHeliosSample.from_dict(
                masked_helios_sample_dict
            )

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = finetuned_model(masked_helios_sample)
                loss = loss_function(logits, label)
                print(f"Loss: {loss.item()}")

            loss.backward()
            adjust_learning_rate(
                optimizer=opt,
                epoch=epoch + (i / len(data_loader)),
                total_epochs=epochs,
                warmup_epochs=int(epochs * 0.1),
                max_lr=lr,
                min_lr=1.0e-5,
            )

            opt.step()
            opt.zero_grad()

    return finetuned_model


config = EvalDatasetConfig(
    task_type=TaskType.CLASSIFICATION,
    imputes=[],
    is_multilabel=False,
    num_classes=10,
    height_width=64,
)

dataloader = DataLoader(
    get_eval_dataset(
        eval_dataset="m-eurosat",
        split="train",
        partition="default",
        norm_stats_from_pretrained=False,
    ),
    collate_fn=eval_collate_fn,
    batch_size=128,
    num_workers=8,
)

train_and_eval_finetune(config, 1e-4, dataloader, torch.device("cuda"), 4, 128, 50)
