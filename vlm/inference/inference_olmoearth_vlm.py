"""
Inference test script for OlmoEarth-VLM Bridge to generate text description of a satellite image.
"""

import torch
import argparse
from PIL import Image
import numpy as np
from pathlib import Path
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vlm.olmoearth_vlm_bridge import OlmoEarthVLMBridge
from olmoearth_pretrain.model_loader import ModelID

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_satellite_image(image_path: str, target_size=(256, 256)) -> torch.Tensor:
    """
    Load and preprocess a satellite image.
    
    Args:
        image_path: Path to image file
        target_size: Target size (H, W)
    
    Returns:
        Preprocessed image tensor [1, C, H, W]
    """
    img = Image.open(image_path).convert('RGB')
    
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5  # Normalize to [-1, 1]
    
    # Convert to tensor [C, H, W]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    
    # Add batch dimension [1, C, H, W]
    tensor = tensor.unsqueeze(0)
    
    # If we have multiple band images, stack them here for up to 12 channels
    if tensor.shape[1] < 12:
        padding = torch.zeros(1, 12 - tensor.shape[1], *target_size)
        tensor = torch.cat([tensor, padding], dim=1)
    
    return tensor


def main():
    parser = argparse.ArgumentParser(description="OlmoEarth-VLM Inference")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to satellite image")
    parser.add_argument("--prompt", type=str, 
                        default="Describe what you see in this satellite image in detail.",
                        help="Text prompt for the model")
    
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    
    model_path = Path(args.model_path)
    if (model_path / "config.json").exists():
        # Load saved model
        model = OlmoEarthVLMBridge.from_pretrained(
            str(model_path),
            device=device,
        )
    else:
        # Create fresh model and optionally load weights
        logger.info("Creating fresh model ")
        model = OlmoEarthVLMBridge(
            olmoearth_model_id=ModelID.OLMOEARTH_V1_NANO,
            vlm_model_id="vikhyatk/moondream2",
            freeze_olmoearth=True,
            freeze_vlm_vision=False,
            use_lora=True,
            fusion_type="geospatial_queries",
            device=device,
        )
    
    model.eval()
    logger.info("Model loaded successfully!")
    logger.info(f"Loading image from {args.image_path}...")
    pixel_values = load_satellite_image(args.image_path).to(device)
    logger.info(f"Image shape: {pixel_values.shape}")
    logger.info(f"Generating description with prompt: '{args.prompt}'")
    
    with torch.no_grad():
        generated_text = model.generate(
            pixel_values=pixel_values,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
  
    print("\n" + "="*80)
    print("GENERATED DESCRIPTION:")
    print("="*80)
    print(generated_text)
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
