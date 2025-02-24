"""Downstream evaluator callback."""

import logging
import time
from dataclasses import dataclass, field

import torch
from olmo_core.eval.evaluator import Evaluator
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader
from upath import UPath

from helios.evals.datasets import GeobenchDataset
from helios.evals.embeddings import get_embeddings
from helios.evals.knn import run_knn

logger = logging.getLogger(__name__)


# Geobench classification
METRIC_NAME = "Top-1 Accuracy"
NAME_PREFIX = "Geobench"
GEOBENCH_DIR = UPath("/weka/dfive-default/presto-geobench/dataset/geobench")


class DownstreamEvaluator:
    """Evaluator for downstream tasks."""

    def __init__(
        self,
        name: str,
        task: str,
        trainer: Trainer,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the downstream evaluator."""
        self.name = name
        self.task = task
        self.trainer = trainer
        self.device = device

    def _get_data_loader(self, split: str) -> DataLoader:
        """Get the data loader for the given split."""
        return DataLoader(
            GeobenchDataset(GEOBENCH_DIR, self.task, split, "default"),
            collate_fn=GeobenchDataset.collate_fn,
        )

    def _get_embeddings(self, data_loader: DataLoader) -> tuple:
        """Get the embeddings for the given data loader."""
        return get_embeddings(
            data_loader=data_loader,
            model=self.trainer.train_module.model.encoder,
            patch_size=self.trainer.train_module.model.encoder.max_patch_size,
        )

    def val(self) -> float:
        """Validate the model on the downstream task."""
        try:
            train_loader = self._get_data_loader("train")
            val_loader = self._get_data_loader("valid")

            train_embeddings, train_labels = self._get_embeddings(train_loader)
            test_embeddings, test_labels = self._get_embeddings(val_loader)

            val_result = run_knn(
                eval_type="KNN-20",
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                num_classes=train_loader.dataset.num_classes,
                is_multilabel=train_loader.dataset.is_multilabel,
                device=self.device,
            )
            logger.info(
                f"Downstream evaluator {self.name} {METRIC_NAME} score: {val_result}"
            )
            return val_result
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return 0


@dataclass
class DownstreamEvaluatorCallback(Callback):
    """Runs in-loop evaluations periodically during training."""

    evaluators: list[DownstreamEvaluator] = field(default_factory=list)
    eval_interval: int = 10
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(10))

    def post_step(self) -> None:
        """Run the evaluators."""
        if self.step <= 1 or self.step % self.eval_interval != 0:
            return

        for evaluator in self.evaluators:
            logger.info(f"Running {evaluator.name} evaluations...")
            start_time = time.monotonic()
            val_result = evaluator.val()
            self.trainer.record_metric(
                f"eval/{evaluator.name}/{METRIC_NAME}", val_result
            )
            logger.info(
                f"Finished {evaluator.name} evaluations in {time.monotonic() - start_time:.1f} seconds."
            )
            logger.info(f"Metric {METRIC_NAME}: {val_result}")


@dataclass
class DownstreamEvaluatorCallbackConfig(CallbackConfig):
    """Config for the downstream evaluator callback."""

    tasks: list[str]
    eval_interval: int = 10
    eval_duration: Duration = field(default_factory=lambda: Duration.epochs(10))
    enabled: bool = True

    def build(self, trainer: Trainer) -> Callback | None:
        """Build the downstream evaluator callback."""
        if not self.enabled:
            return None

        evaluators: list[Evaluator] = [
            DownstreamEvaluator(
                name=f"{NAME_PREFIX}-{task}",
                task=task,
                trainer=trainer,
                device=trainer.device,
            )
            for task in self.tasks
        ]

        return DownstreamEvaluatorCallback(
            evaluators=evaluators,
            eval_interval=self.eval_interval,
            eval_duration=self.eval_duration,
        )
