
import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from PIL import Image
import numpy as np

class ChatEarthNetDataset(Dataset):
    def __init__(self, data_dir, split="train", tokenizer_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", use_full_dataset=True):
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Use ChatEarthNet_caps_35 (98K samples)
        dataset_version = "35" if use_full_dataset else "4v"
        json_path = os.path.join(data_dir, "json_files", f"ChatEarthNet_caps_{dataset_version}_{split}.json")
        
        if not os.path.exists(json_path):
            json_path = os.path.join(data_dir, "json_files", f"ChatEarthNet_caps_4v_{split}.json")
            print(f"Note: Using 4v dataset (fallback)")
        
     
        print(f"Loading {split} data from {json_path}...")
        with open(json_path, 'r') as f:
            self.samples = json.load(f)
        
        # Image directories (RGB + extra Sentinel-2 bands)
        self.rgb_dir = os.path.join(data_dir, "s2_rgb_images", "s2_images")
        self.band567_dir = os.path.join(data_dir, "s2_band_5_6_7_images")
        self.band81112_dir = os.path.join(data_dir, "s2_band_8_11_12_images")

        for p in [self.rgb_dir, self.band567_dir, self.band81112_dir]:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image directory not found: {p}")
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print("Images will be loaded from:")
        print(f"  RGB:     {self.rgb_dir}")
        print(f"  Bands5-7:{self.band567_dir}")
        print(f"  Bands8/11/12:{self.band81112_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image_name = sample.get("image", sample.get("image_id"))

        def load_three_channel(path: str) -> torch.Tensor:
            if not os.path.exists(path):
                return None
            img = Image.open(path).convert('RGB')
            if img.size != (256, 256):
                img = img.resize((256, 256), Image.Resampling.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            return tensor

        rgb_path = os.path.join(self.rgb_dir, image_name)
        b567_path = os.path.join(self.band567_dir, image_name)
        b81112_path = os.path.join(self.band81112_dir, image_name)

        rgb = load_three_channel(rgb_path)
        b567 = load_three_channel(b567_path)
        b81112 = load_three_channel(b81112_path)

        if rgb is None or b567 is None or b81112 is None:
            print(f"Warning: Missing bands for {image_name}, falling back to random tensor")
            pixel_values = torch.randn(12, 256, 256)
        else:
            bands = [rgb, b567, b81112]  # 9 channels
            pixel_values = torch.cat(bands, dim=0)  # [9, H, W]
            # Pad to 12 channels if needed
            if pixel_values.shape[0] < 12:
                pad = torch.zeros(12 - pixel_values.shape[0], 256, 256)
                pixel_values = torch.cat([pixel_values, pad], dim=0)
        
        caption = sample.get("caption", "A satellite image.")
        
        # Ensure caption is a string (handle if it's a list)
        if isinstance(caption, list):
            caption = " ".join(caption) if caption else "A satellite image."
        
        # Format text without chat template 
        # Simple format: "Question: Describe this satellite image. Answer: {caption}"
        text = f"Question: Describe this satellite image.\nAnswer: {caption}"
        
        # Tokenize
        encodings = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=256, truncation=True)
        input_ids = encodings.input_ids.squeeze(0)
        attention_mask = encodings.attention_mask.squeeze(0)
        labels = input_ids.clone()
        
        # Mask out non-assistant tokens in labels
        assistant_start_token = "<|im_start|>assistant\n"
        assistant_start_idx = text.find(assistant_start_token)
        if assistant_start_idx != -1:
            prefix_text = text[:assistant_start_idx + len(assistant_start_token)]
            prefix_encodings = self.tokenizer(prefix_text, return_tensors="pt")
            prefix_len = prefix_encodings.input_ids.shape[1]
            # Set labels for prefix to -100 (ignored by cross entropy)
            labels[:prefix_len] = -100
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
