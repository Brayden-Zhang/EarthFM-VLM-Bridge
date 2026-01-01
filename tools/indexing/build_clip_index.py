#!/usr/bin/env python3
"""
Build a FAISS index using CLIP image embeddings from actual satellite images.
This enables true visual semantic search - finding images by what they look like,
not just by their text descriptions.
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

JSON_DIR = "chatearthnet_data/json_files"
IMAGE_DIR = "chatearthnet_data/s2_rgb_images/s2_images"
OUTPUT_DIR = "clip_search_index"
BATCH_SIZE = 32  # Smaller batches for GPU memory
CLIP_MODEL = "openai/clip-vit-base-patch32"  # Good balance of speed/quality

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

def build_clip_index(data, device="cuda"):
    """Build FAISS index using CLIP image embeddings."""
    print(f"Loading CLIP model ({CLIP_MODEL})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    
    if torch.cuda.is_available() and device == "cuda":
        model = model.to("cuda")
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")
    
    model.eval()
    
    print(f"Embedding {len(data)} images in batches of {BATCH_SIZE}...")
    
    all_embeddings = []
    failed_indices = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(data), BATCH_SIZE), desc="Embedding images"):
            batch_data = data[i:i+BATCH_SIZE]
            batch_images = []
            batch_indices = []
            
            # Load images in batch
            for j, item in enumerate(batch_data):
                try:
                    img = Image.open(item["image_path"]).convert("RGB")
                    batch_images.append(img)
                    batch_indices.append(i + j)
                except Exception as e:
                    failed_indices.append(i + j)
                    continue
            
            if not batch_images:
                continue
            
            # Process and embed
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            if device == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            outputs = model.get_image_features(**inputs)
            embeddings = outputs.cpu().numpy()
            
            all_embeddings.append(embeddings)
    
    if failed_indices:
        print(f"Warning: {len(failed_indices)} images failed to load")
        # Remove failed items from data
        data = [item for i, item in enumerate(data) if i not in set(failed_indices)]
    
    # Concatenate all embeddings
    embeddings_matrix = np.vstack(all_embeddings).astype('float32')
    print(f"Embeddings shape: {embeddings_matrix.shape}")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_matrix)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings_matrix.shape[1]
    
    # Use IVF index for faster search on large datasets
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
    
    return index, embeddings_matrix, data

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load all image paths
    data = load_all_image_paths()
    
    # Filter to existing images
    data = filter_existing_images(data)
    
    if len(data) == 0:
        print("No images found! Check paths.")
        return
    
    # Build CLIP index
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index, embeddings, filtered_data = build_clip_index(data, device)
    
    # Save everything
    print("Saving index and metadata...")
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss_index.bin"))
    
    # Save metadata
    metadata = {
        "items": filtered_data,
        "total_count": len(filtered_data),
        "model": CLIP_MODEL,
        "embedding_type": "image"
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    # Save embeddings
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    
    print(f"\nCLIP Image Index built successfully!")
    print(f"  - FAISS index: {OUTPUT_DIR}/faiss_index.bin")
    print(f"  - Metadata: {OUTPUT_DIR}/metadata.pkl")
    print(f"  - Embeddings: {OUTPUT_DIR}/embeddings.npy")
    print(f"  - Total images indexed: {len(filtered_data)}")

if __name__ == "__main__":
    main()
