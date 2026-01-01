"""
Training script for OlmoEarth-VLM Bridge on ChatEarthNet dataset.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
import logging
from tqdm import tqdm
from typing import Optional
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlm.olmoearth_vlm_bridge import OlmoEarthVLMBridge
from dataset import ChatEarthNetDataset
from olmoearth_pretrain.model_loader import ModelID

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
    epoch: int,
    accumulation_steps: int = 4,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(progress_bar):
        # Move to device
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
        
        # Track loss
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item() * accumulation_steps,
            'avg_loss': total_loss / num_batches,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train OlmoEarth-VLM Bridge")
    
    # Model arguments
    parser.add_argument("--olmoearth_model", type=str, default="OLMOEARTH_V1_NANO",
                        help="OlmoEarth model ID (NANO, TINY, BASE, LARGE)")
    parser.add_argument("--vlm_model", type=str, default="vikhyatk/moondream2",
                        help="VLM model ID")
    parser.add_argument("--fusion_type", type=str, default="geospatial_queries",
                        choices=["cross_attention", "geospatial_queries"],
                        help="Type of fusion module")
    parser.add_argument("--num_geo_queries", type=int, default=32,
                        help="Number of geospatial queries")
    parser.add_argument("--freeze_olmoearth", action="store_true", default=True,
                        help="Freeze OlmoEarth encoder")
    parser.add_argument("--freeze_vlm_vision", action="store_true", default=False,
                        help="Freeze VLM vision encoder")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA for VLM")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    
    # Training arguments
    parser.add_argument("--data_dir", type=str, default="./chatearthnet_data",
                        help="Path to ChatEarthNet dataset")
    parser.add_argument("--output_dir", type=str, default="./olmoearth_vlm_checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--save_every", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--validate_every", type=int, default=200, help="Validate every N steps")
    
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_dataset = ChatEarthNetDataset(
        data_dir=args.data_dir,
        split="train",
        tokenizer_id=args.vlm_model,
        use_full_dataset=True,  
    )
    
    val_dataset = ChatEarthNetDataset(
        data_dir=args.data_dir,
        split="val",
        tokenizer_id=args.vlm_model,
        use_full_dataset=True,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    
    logger.info("Creating model...")
    
    olmoearth_model_map = {
        "OLMOEARTH_V1_NANO": ModelID.OLMOEARTH_V1_NANO,
        "OLMOEARTH_V1_TINY": ModelID.OLMOEARTH_V1_TINY,
        "OLMOEARTH_V1_BASE": ModelID.OLMOEARTH_V1_BASE,
        "OLMOEARTH_V1_LARGE": ModelID.OLMOEARTH_V1_LARGE,
    }
    
    olmoearth_model_id = olmoearth_model_map.get(args.olmoearth_model.upper(), ModelID.OLMOEARTH_V1_NANO)
    
    model = OlmoEarthVLMBridge(
        olmoearth_model_id=olmoearth_model_id,
        vlm_model_id=args.vlm_model,
        freeze_olmoearth=args.freeze_olmoearth,
        freeze_vlm_vision=args.freeze_vlm_vision,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        fusion_type=args.fusion_type,
        num_geo_queries=args.num_geo_queries,
        device=device,
    )
    
    logger.info("Setting up optimizer...")
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = AdamW(
        trainable_params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(train_loader) * args.num_epochs // args.accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    logger.info(f"Total training steps: {total_steps}")
    
    logger.info("Starting training...")
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch + 1,
            accumulation_steps=args.accumulation_steps,
        )
        
        logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % 2 == 0 or epoch == args.num_epochs - 1:
            val_loss = validate(
                model=model,
                dataloader=val_loader,
                device=device,
            )
            
            logger.info(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.output_dir, "best_model")
                model.save_pretrained(save_path)
                logger.info(f"Saved best model to {save_path}")
        
        if (epoch + 1) % 2 == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}")
            model.save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")
    
    final_path = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_path)
    logger.info(f"Saved final model to {final_path}")
    logger.info("Training done.")


if __name__ == "__main__":
    main()
