#!/usr/bin/env python3
"""
Build a FAISS index of ALL 163K+ ChatEarthNet images using pre-computed captions.
"""

import json
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

JSON_DIR = "chatearthnet_data/json_files"
IMAGE_DIR = "chatearthnet_data/s2_rgb_images/s2_images"
OUTPUT_DIR = "semantic_search_index"
BATCH_SIZE = 512

def load_all_captions():
    """Load all captions from JSON files."""
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
                # Normalize different formats
                image = item.get("image") or item.get("image_id")
                caption = item.get("caption")
                # Handle caption as list in some files
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

def build_faiss_index(data):
    """Build FAISS index from captions."""
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Extract captions
    captions = [item["caption"] for item in data]
    
    print(f"Embedding {len(captions)} captions in batches of {BATCH_SIZE}...")
    
    # Embed in batches
    all_embeddings = []
    for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="Embedding"):
        batch = captions[i:i+BATCH_SIZE]
        embeddings = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embeddings)
    
    # Concatenate all embeddings
    embeddings_matrix = np.vstack(all_embeddings).astype('float32')
    print(f"Embeddings shape: {embeddings_matrix.shape}")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_matrix)
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings_matrix.shape[1]
    
    if len(data) > 10000:
        nlist = min(1000, len(data) // 100)  # Number of clusters
        quantizer = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity calculation
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        print(f"Training IVF index with {nlist} clusters...")
        index.train(embeddings_matrix)
        index.nprobe = 50  # Number of clusters to search
    else:
        index = faiss.IndexFlatIP(dimension)  # Flat index for smaller datasets
    
    index.add(embeddings_matrix)
    print(f"Index contains {index.ntotal} vectors")
    
    return index, embeddings_matrix

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    data = load_all_captions()
    
    data = filter_existing_images(data)
    
    if len(data) == 0:
        print("No images found! Check paths.")
        return
    
    index, embeddings = build_faiss_index(data)
    
    
    faiss.write_index(index, os.path.join(OUTPUT_DIR, "faiss_index.bin"))
    
    metadata = {
        "items": data,
        "total_count": len(data)
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    # Save embeddings for potential re-use
    np.save(os.path.join(OUTPUT_DIR, "embeddings.npy"), embeddings)
    
    print(f"\nIndex built successfully!")
    print(f"  - FAISS index: {OUTPUT_DIR}/faiss_index.bin")
    print(f"  - Metadata: {OUTPUT_DIR}/metadata.pkl")
    print(f"  - Embeddings: {OUTPUT_DIR}/embeddings.npy")
    print(f"  - Total images indexed: {len(data)}")

if __name__ == "__main__":
    main()
