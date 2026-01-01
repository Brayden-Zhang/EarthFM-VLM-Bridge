#!/usr/bin/env python3
"""
Build a FAISS index using OlmoEarth-VLM geospatial embeddings.
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import faiss

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration
JSON_DIR = "chatearthnet_data/json_files"
IMAGE_DIR = "chatearthnet_data/s2_rgb_images/s2_images"
OUTPUT_DIR = "olmoearth_search_index"
BATCH_SIZE = 32
CHECKPOINT_DIR = "vlm_full_checkpoints/best_model"


def load_olmoearth_encoder():
    """Load the OlmoEarth encoder from the trained VLM checkpoint."""
    from olmoearth_pretrain.model_loader import load_model_from_id, ModelID
    
    encoder = load_model_from_id(ModelID.OLMOEARTH_V1_NANO)
    encoder.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    
    return encoder, device


def load_fusion_module():
    """Load the trained fusion module that projects OlmoEarth features."""
    fusion_path = os.path.join(CHECKPOINT_DIR, "fusion.pt")
    
    if os.path.exists(fusion_path):
        print(f"Loading trained fusion module from {fusion_path}")
        from vlm.olmoearth_vlm_bridge import GeospatialQueryEncoder
        
        # Load config
        config_path = os.path.join(CHECKPOINT_DIR, "config.json")
        with open(config_path) as f:
            config = json.load(f)
        
        # Get actual output_dim from saved weights
        state_dict = torch.load(fusion_path, map_location="cpu")
        output_dim = state_dict["geo_queries"].shape[1]  # [num_queries, output_dim]
        print(f"  Fusion output dim from checkpoint: {output_dim}")
        
        fusion = GeospatialQueryEncoder(
            num_queries=32,
            olmoearth_dim=config["olmoearth_dim"],
            output_dim=output_dim
        )
        fusion.load_state_dict(state_dict)
        fusion.eval()
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fusion = fusion.to(device)
        
        return fusion, output_dim
    else:
        print("No fusion module found, will use raw OlmoEarth features")
        return None, 128


def preprocess_image_for_olmoearth(img_path: str, device: str):

    from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample
    
    img = Image.open(img_path).convert("RGB")
    
    # Resize to expected size (64x64 for nano model)
    img = img.resize((64, 64), Image.BILINEAR)
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # OlmoEarth expects [B, H, W, T, C] format for sentinel2_l2a
    # T = number of timestamps, we'll use 1
    # For RGB images, we'll replicate to simulate multi-band
    H, W, C = img_array.shape
    
    # Create tensor [1, H, W, 1, C] 
    pixel_values = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(3)  # [1, H, W, 1, C]
    pixel_values = pixel_values.to(device)
    
    # Create random metadata since chatearthnet does not provide it
    timestamps = torch.zeros((1, 1, 3), device=device, dtype=torch.long)
    latlon = torch.zeros((1, 2), device=device)
    
    # mask is zeros since we have full data
    num_band_sets = 3
    mask_shape = (1, H, W, 1, num_band_sets)
    sentinel2_l2a_mask = torch.zeros(mask_shape, device=device, dtype=torch.long)
    
    sample = MaskedOlmoEarthSample(
        timestamps=timestamps,
        latlon=latlon,
        sentinel2_l2a=pixel_values,
        sentinel2_l2a_mask=sentinel2_l2a_mask
    )
    
    return sample


def extract_features_batch(encoder, fusion, image_paths: list, device: str):
    """Extract features for a batch of images."""
    batch_features = []
    
    for img_path in image_paths:
        try:
            sample = preprocess_image_for_olmoearth(img_path, device)
            
            with torch.no_grad():
                # Extract OlmoEarth features
                output = encoder(sample, patch_size=8)
                
                # Get the latent features
                if isinstance(output, tuple) and len(output) >= 1:
                    latent = output[0]
                    
                    # Extract sentinel2_l2a features
                    if hasattr(latent, 'sentinel2_l2a') and latent.sentinel2_l2a is not None:
                        features = latent.sentinel2_l2a
                    else:
                        for field in latent._fields:
                            if not field.endswith('_mask'):
                                val = getattr(latent, field, None)
                                if val is not None and isinstance(val, torch.Tensor):
                                    features = val
                                    break
                else:
                    features = output
                
                # Reshape to [B, N, D]
                if features.ndim == 6:
                    B, P_H, P_W, T, Band_Sets, D = features.shape
                    features = features.reshape(B, -1, D)
                elif features.ndim == 5:
                    B, P_H, P_W, T, D = features.shape
                    features = features.reshape(B, -1, D)
                elif features.ndim == 4:
                    B, P_H, P_W, D = features.shape
                    features = features.reshape(B, -1, D)
                
                # If we have a fusion module, use it for better embeddings
                if fusion is not None:
                    features = fusion(features)  # [B, num_queries, output_dim]
                
                # Pool to single vector per image
                # Mean pooling over spatial dimension
                pooled = features.mean(dim=1)  # [B, D]
                
                batch_features.append(pooled.cpu().numpy())
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Return zero embedding for failed images
            embed_dim = 2048 if fusion is not None else 128
            batch_features.append(np.zeros((1, embed_dim), dtype=np.float32))
    
    return np.vstack(batch_features) if batch_features else None


def load_all_image_paths():
    """Load all image paths from JSON files."""
    files = [
        "ChatEarthNet_caps_35_train.json",
        "ChatEarthNet_caps_35_val.json", 
        "ChatEarthNet_caps_35_test.json"
    ]
    
    all_data = []
    for filename in files:
        filepath = os.path.join(JSON_DIR, filename)
        print(f"Loading {filename}...")
        with open(filepath) as f:
            data = json.load(f)
            for item in data:
                image = item.get("image") or item.get("image_id")
                caption = item.get("caption")
                if isinstance(caption, list):
                    caption = caption[0] if caption else ""
                all_data.append({
                    "image": image,
                    "caption": caption
                })
    
    print(f"Total loaded: {len(all_data)} items")
    return all_data


def filter_existing_images(data):
    """Filter to only images that exist on disk."""
    existing = []
    missing = 0
    
    for item in tqdm(data, desc="Checking images"):
        image_path = os.path.join(IMAGE_DIR, item["image"])
        if os.path.exists(image_path):
            existing.append({
                "image_id": item["image"],
                "caption": item["caption"],
                "image_path": image_path
            })
        else:
            missing += 1
    
    print(f"Found {len(existing)} existing images, {missing} missing")
    return existing


def build_olmoearth_index(data):
    """Build FAISS index using OlmoEarth embeddings."""
    encoder, device = load_olmoearth_encoder()
    fusion, embed_dim = load_fusion_module()
    
    print(f"Embedding dimension: {embed_dim}")
    print(f"Embedding {len(data)} images in batches of {BATCH_SIZE}...")
    
    all_embeddings = []
    failed_indices = []
    
    for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Embedding images"):
        batch_data = data[i:i+BATCH_SIZE]
        image_paths = [item["image_path"] for item in batch_data]
        
        embeddings = extract_features_batch(encoder, fusion, image_paths, device)
        
        if embeddings is not None:
            all_embeddings.append(embeddings)
        
        # Clear GPU cache periodically
        if i % (BATCH_SIZE * 10) == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Concatenate all embeddings
    embeddings_matrix = np.vstack(all_embeddings).astype('float32')
    print(f"Embeddings shape: {embeddings_matrix.shape}")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_matrix)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings_matrix.shape[1]
    
    if len(data) > 10000:
        nlist = min(1000, len(data) // 100)
        quantizer = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        print(f"Training IVF index with {nlist} clusters...")
        index.train(embeddings_matrix)
        index.nprobe = 50
    else:
        index = faiss.IndexFlatIP(dimension)
    
    index.add(embeddings_matrix)
    print(f"Index contains {index.ntotal} vectors")
    
    return index, embeddings_matrix


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data = load_all_image_paths()
    
    data = filter_existing_images(data)
     
    index, embeddings = build_olmoearth_index(data)
       
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss_index.bin"))
    
    metadata = {
        "items": data,
        "total_count": len(data),
        "model": "olmoearth_vlm",
        "embedding_type": "geospatial"
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    
    print(f"\nOlmoEarth Image Index built successfully!")
    print(f"  - FAISS index: {OUTPUT_DIR}/faiss_index.bin")
    print(f"  - Metadata: {OUTPUT_DIR}/metadata.pkl")
    print(f"  - Embeddings: {OUTPUT_DIR}/embeddings.npy")
    print(f"  - Total images indexed: {len(data)}")


if __name__ == "__main__":
    main()
