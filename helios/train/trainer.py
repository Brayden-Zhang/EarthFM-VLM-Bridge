"""Trainer based on olmo_core."""

from logging import getLogger
from typing import Any

import torch
from olmo_core.train.trainer import Trainer
from olmo_core.utils import move_to_device

logger = getLogger(__name__)


class HeliosTrainer(Trainer):
    """Trainer for Helios."""

    def model_forward(self, micro_batch: dict[str, Any]) -> torch.Tensor:
        """Run a forward pass on a micro-batch, returning the logits."""
        raise NotImplementedError("model forward helios")

    def get_losses(
        self,
    ) -> dict[str, Any]:
        """Compute the losses for a micro-batch and logits."""
        raise NotImplementedError("get losses helios")

    def eval_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Evaluate a batch."""
        raise NotImplementedError("eval batch helios")

    def _train_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Train a batch."""
        # Record how many instances are going to be skipped (masked out).
        # if (instance_mask := batch.get("instance_mask")) is not None and not dry_run:
        #     self.record_metric("train/masked instances", (~instance_mask).sum(), ReduceType.sum)

        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Generate labels, calculate how many tokens are going to be use in the loss.
        # if "labels" not in batch:
        #     batch["labels"] = self._get_labels(batch)
        # batch_num_tokens_for_loss = (
        #     batch["labels"] != self.data_loader.collator.label_ignore_index
        # ).sum()

        # Split into micro-batches.
        if self.rank_microbatch_size < (seq_len := batch["input_ids"].shape[1]):
            raise RuntimeError(
                f"Microbatch size ({self.rank_microbatch_size}) is too small relative to sequence length ({seq_len})"
            )

        # ce_batch_loss = move_to_device(torch.tensor(0.0), self.device)
        # z_batch_loss = (
        #     None
        #     if self.z_loss_multiplier is None
        #     else move_to_device(torch.tensor(0.0), self.device)
        # )

        # NO MICROBATCHING for now
        logits = self.model_forward(batch)
        # TODO: Get loss
        # In case this helps with memory utilization.
        del batch

        # Need to record loss metrics
        # if dry_run:
        #     # Zero-gradients again.
        #     self.optim.zero_grad(set_to_none=True)
        #     return

        # self.record_metric(TRAIN_CE_LOSS_METRIC, ce_batch_loss, ReduceType.mean)
        # if z_batch_loss is not None:
        #     self.record_metric(TRAIN_Z_LOSS_METRIC, z_batch_loss, ReduceType.mean)

        # if isinstance(self.optim, SkipStepOptimizer):
        #     self.optim.latest_loss = ce_batch_loss

        # Run through callbacks.
        for callback in self.callbacks.values():
            callback.pre_optim_step()

        # Optimizer step.
        self.optim.step()
        # if isinstance(self.optim, SkipStepOptimizer):
        #     self.record_metric(OPTIM_STEP_SKIPPED_METRIC, self.optim.step_skipped)

        # Run through callbacks.
        for callback in self.callbacks.values():
            callback.post_train_batch()

    def _dry_run_batch(self):
        logger.info("dry run batch helios")
        return {}

    def _fit_epoch(self):
        """Copied almost directly from olmo_core.train.trainer.Trainer._fit_epoch

        but removing the seq len logging metric collection
        """
        self.data_loader.reshuffle(self.epoch)

        logger.info(f"Starting epoch {self.epoch}...")

        for callback in self.callbacks.values():
            callback.pre_epoch()

        first_batch = True
        for batch in self._iter_batches():
            # Bookkeeping.
            self.global_step += 1
            # self.global_train_tokens_seen += self._validate_batch(batch)

            # self.record_metric(SEQ_LEN_METRIC, float(batch["input_ids"].shape[1]))

            for callback in self.callbacks.values():
                callback.pre_step(batch)

            self._train_batch(batch)

            for callback in self.callbacks.values():
                callback.post_step()

            if first_batch or self.global_step % self.metrics_collect_interval == 0:
                self._log_metrics()
                if torch.cuda.is_available():
                    torch.cuda.set_sync_debug_mode("warn")

            first_batch = False

            if self.training_complete:
                # Finishing before the epoch is complete.
                # Log any remaining metrics.
                self._log_metrics()
                return

        # Log any remaining metrics.
        self._log_metrics()

        logger.info("Epoch complete")

        for callback in self.callbacks.values():
            callback.post_epoch()

        # Bookkeeping
        self.epoch += 1
        self.data_loader.reset()
