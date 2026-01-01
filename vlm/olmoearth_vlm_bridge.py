"""
OlmoEarth + VLM Bridge Architecture
Integrates OlmoEarth's pretrained geospatial representations with the Moondream2 VLM
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, Dict, Any, Union, Tuple
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from olmoearth_pretrain.model_loader import load_model_from_id, ModelID
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample

logger = logging.getLogger(__name__)


class GeospatialQueryEncoder(nn.Module):
    """
    Encodes OlmoEarth features into a set of learnable geospatial queries.
    Based on the Perceiver model.
    """
    def __init__(self, num_queries: int = 32, olmoearth_dim: int = 128, output_dim: int = 1152):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable geospatial query embeddings
        self.geo_queries = nn.Parameter(torch.randn(num_queries, output_dim) * 0.02)
        
        # Project OlmoEarth features
        self.olmo_proj = nn.Linear(olmoearth_dim, output_dim)
        self.ln_olmo = nn.LayerNorm(output_dim)
        self.ln_queries = nn.LayerNorm(output_dim)
        
        # Multi-head attention for queries to attend to OlmoEarth features
        self.attn = nn.MultiheadAttention(output_dim, num_heads=8, batch_first=True)
        
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 4, output_dim),
        )
        self.ln_ffn = nn.LayerNorm(output_dim)
        
    def forward(self, olmo_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            olmo_features: [B, N, olmoearth_dim]
        Returns:
            geo_queries: [B, num_queries, output_dim]
        """
        batch_size = olmo_features.shape[0]
        
        # Project OlmoEarth features
        olmo_proj = self.ln_olmo(self.olmo_proj(olmo_features))
        
        # Expand queries to batch
        queries = repeat(self.geo_queries, 'n d -> b n d', b=batch_size)
        queries = self.ln_queries(queries)
        
        attn_out, _ = self.attn(queries, olmo_proj, olmo_proj)
        queries = queries + attn_out
        
        ffn_out = self.ffn(queries)
        queries = self.ln_ffn(queries + ffn_out)
        
        return queries


class OlmoEarthVLMBridge(nn.Module):
    """
    Main bridge architecture combining OlmoEarth with Moondream2 VLM.
    
    Architecture:
    1. OlmoEarth encoder: Extracts rich geospatial features from satellite imagery
    2. GeospatialQueryEncoder: Encodes features into learnable geospatial queries
    3. Moondream2 language model: Generates text conditioned on geospatial queries
    """
    
    def __init__(
        self,
        olmoearth_model_id: ModelID = ModelID.OLMOEARTH_V1_NANO,
        vlm_model_id: str = "vikhyatk/moondream2",
        freeze_olmoearth: bool = True,
        freeze_vlm_vision: bool = False,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        num_geo_queries: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        
        self.device = device
        
        logger.info(f"Initializing OlmoEarth-VLM Bridge")
        logger.info(f"  OlmoEarth: {olmoearth_model_id}")
        logger.info(f"  VLM: {vlm_model_id}")
        
        logger.info("Loading OlmoEarth encoder...")
        self.olmoearth_encoder = load_model_from_id(olmoearth_model_id)
        
        if hasattr(self.olmoearth_encoder, 'encoder') and hasattr(self.olmoearth_encoder.encoder, 'norm'):
            self.olmoearth_dim = self.olmoearth_encoder.encoder.norm.normalized_shape[0]
        else:
            self.olmoearth_dim = 128  # Default for nano
        
        logger.info(f"  OlmoEarth output dim: {self.olmoearth_dim}")
        
        if freeze_olmoearth:
            for param in self.olmoearth_encoder.parameters():
                param.requires_grad = False
            logger.info("  OlmoEarth encoder frozen")
        else:
            logger.info("  OlmoEarth encoder trainable")
        
        # Load Moondream2 VLM
        logger.info(f"Loading Moondream2 VLM on {device}...")
        
        load_device = device if device == "gpu" else "cpu"
        
        self.vlm = AutoModelForCausalLM.from_pretrained(
            vlm_model_id,
            trust_remote_code=True,
            revision="2024-08-26",
            torch_dtype=torch.float32, 
            device_map=load_device if load_device == "gpu" else "cpu",
            low_cpu_mem_usage=True,
        )
     
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(vlm_model_id, trust_remote_code=True)
        
        # Get VLM vision encoder dimension
        # Moondream2 uses SigLIP vision encoder with output dim 1152
        self.vlm_vision_dim = 1152
        logger.info(f"  VLM vision dim: {self.vlm_vision_dim}")
        
        # Get text model embedding dimension for projection
        base_model = self.vlm.get_base_model() if hasattr(self.vlm, 'get_base_model') else self.vlm
        text_model = base_model.text_model if hasattr(base_model, 'text_model') else base_model
        self.text_embed_dim = text_model.get_input_embeddings().embedding_dim
        logger.info(f"  Text embedding dim: {self.text_embed_dim}")
        
        # 3. Create geospatial query encoder - output directly to text embedding space
        logger.info(f"Creating GeospatialQueryEncoder with {num_geo_queries} queries")
        self.fusion = GeospatialQueryEncoder(
            num_queries=num_geo_queries,
            olmoearth_dim=self.olmoearth_dim,
            output_dim=self.text_embed_dim
        )
        
        if use_lora:
            logger.info(f"Applying LoRA to language model (r={lora_r}, alpha={lora_alpha})")
            
            # Moondream2 uses Phi architecture with different module names
            # Target fc1, fc2 in MLP layers and attention projection layers
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["fc1", "fc2", "q_proj", "k_proj", "v_proj", "dense"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=None,
            )
            
            # Apply LoRA
            
            self.vlm = get_peft_model(self.vlm, lora_config)
            self.vlm.print_trainable_parameters()
         
        
        # Move models to device
        self.olmoearth_encoder = self.olmoearth_encoder.to(device)
        self.fusion = self.fusion.to(device)
        self.vlm = self.vlm.to(device)
        
        logger.info("OlmoEarth-VLM Bridge initialized successfully!")
    
    def extract_olmoearth_features(self, pixel_values: Union[torch.Tensor, Dict]) -> torch.Tensor:
        """
        Extract features from OlmoEarth encoder.
        
        Args:
            satellite imagery (B, C, H, W)
        
        Returns:
            features: [B, N, olmoearth_dim] 
        """
        # Prepare input for OlmoEarth
        if isinstance(pixel_values, torch.Tensor):
            # Convert [B, C, H, W] -> [B, H, W, 1, C]
            if pixel_values.ndim == 4:
                pixel_values = rearrange(pixel_values, 'b c h w -> b h w 1 c')

            # convert [B, C, T, H, W] -> [B, H, W, T, C]
            elif pixel_values.ndim == 5:
                pixel_values = rearrange(pixel_values, 'b c t h w -> b h w t c')
            
            batch_size = pixel_values.shape[0]
            device = pixel_values.device
            
            # Create rand metadata since chatearthnet dataset does not provide it
            timestamps = torch.zeros((batch_size, 1, 3), device=device, dtype=torch.long)
            latlon = torch.zeros((batch_size, 2), device=device)
            
            # Create mask 
            num_band_sets = 3  # using only rgb for now
            mask_shape = (batch_size, pixel_values.shape[1], pixel_values.shape[2], pixel_values.shape[3], num_band_sets)
            sentinel2_l2a_mask = torch.zeros(mask_shape, device=device, dtype=torch.long)
            
            sample = MaskedOlmoEarthSample(
                timestamps=timestamps,
                latlon=latlon,
                sentinel2_l2a=pixel_values,
                sentinel2_l2a_mask=sentinel2_l2a_mask
            )
        else:
            sample = pixel_values
        
        # Extract features
        # OlmoEarth needs patch_size parameter (use 8 for nano model)
        patch_size = 8
        
        with torch.set_grad_enabled(not self.olmoearth_encoder.training or self.training):
            output = self.olmoearth_encoder(sample, patch_size=patch_size)
        
        # Get encoder output
        # OlmoEarth LatentMIM returns a tuple: (latent, decoded, projected_pooled, reconstructed, metadata)
        # latent is a TokensAndMasks object containing features for each modality
        if isinstance(output, tuple) and len(output) >= 1:
            latent = output[0]  # TokensAndMasks object
            
            # Extract the sentinel2_l2a features (primary features for our use case)
            if isinstance(latent, tuple) and hasattr(latent, '_fields'):  # NamedTuple
                # Get sentinel2_l2a features
                if hasattr(latent, 'sentinel2_l2a') and latent.sentinel2_l2a is not None:
                    features = latent.sentinel2_l2a
                else:
                    # Try to find any non-None modality
                    for field in latent._fields:
                        if not field.endswith('_mask'):
                            val = getattr(latent, field, None)
                            if val is not None and isinstance(val, torch.Tensor):
                                features = val
                                break
            else:
                features = latent
        elif isinstance(output, dict) and 'encoder_out' in output:
            features = output['encoder_out']
        else:
            features = output
        
        # Features from OlmoEarth are typically [B, P_H, P_W, T, Band_Sets, D]
        # We need to reshape to [B, N, D] for our fusion module
        if features.ndim == 6:
            # Reshape [B, P_H, P_W, T, Band_Sets, D] -> [B, P_H*P_W*T*Band_Sets, D]
            B, P_H, P_W, T, Band_Sets, D = features.shape
            features = features.reshape(B, P_H * P_W * T * Band_Sets, D)
        elif features.ndim == 5:
            # Reshape [B, P_H, P_W, T, D] -> [B, P_H*P_W*T, D]
            B, P_H, P_W, T, D = features.shape
            features = features.reshape(B, P_H * P_W * T, D)
        elif features.ndim == 4:
            # Reshape [B, P_H, P_W, D] -> [B, P_H*P_W, D]
            B, P_H, P_W, D = features.shape
            features = features.reshape(B, P_H * P_W, D)
        # else features is already [B, N, D]
        
        return features
    
    def forward(
        self,
        pixel_values: Union[torch.Tensor, Dict],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the OlmoEarth-VLM bridge.
        
        Args:
            pixel_values: Satellite imagery for OlmoEarth [B, C, H, W]
            input_ids: Text input token IDs [B, L]
            attention_mask: Attention mask for text [B, L]
            labels: Labels for language modeling [B, L]
        
        Returns:
            Dictionary containing loss and logits
        """
        batch_size = input_ids.shape[0]
        
        olmo_features = self.extract_olmoearth_features(pixel_values)  # [B, N_olmo, D_olmo]
                
        fused_features = self.fusion(olmo_features)  # [B, num_queries, text_embed_dim]
        
     
        if labels is not None:
            # Access the text model directly for training
            base_model = self.vlm.get_base_model() if hasattr(self.vlm, 'get_base_model') else self.vlm
            text_model = base_model.text_model if hasattr(base_model, 'text_model') else base_model
            
            try:
                # Get text embeddings
                text_embeds = text_model.get_input_embeddings()(input_ids)  # [B, L, D]
                
                # Fused features are already in text_embed_dim space
                # Combine image embeddings with text embeddings: [image_embeds, text_embeds]
                combined_embeds = torch.cat([fused_features, text_embeds], dim=1)  # [B, num_queries+L, D]
                
                # Update attention mask to include image tokens
                batch_size, num_img_tokens = fused_features.shape[:2]
                img_attention = torch.ones(batch_size, num_img_tokens, device=attention_mask.device, dtype=attention_mask.dtype)
                combined_attention_mask = torch.cat([img_attention, attention_mask], dim=1)
                
                # Update labels: -100 for image tokens (don't compute loss on them)
                img_labels = torch.full((batch_size, num_img_tokens), -100, device=labels.device, dtype=labels.dtype)
                combined_labels = torch.cat([img_labels, labels], dim=1)
                
                # Forward through text model with combined embeddings
                outputs = text_model(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attention_mask,
                    labels=combined_labels,
                    use_cache=False,
                )
            except Exception as e:
                logger.warning(f"Error calling text_model: {e}")
                import traceback
                traceback.print_exc()
                # Return dummy outputs for testing
                batch_size, seq_len = input_ids.shape
                vocab_size = 51200  # Moondream2 vocab size
                return {
                    "loss": torch.tensor(0.0, device=input_ids.device, requires_grad=True),
                    "logits": torch.randn(batch_size, seq_len, vocab_size, device=input_ids.device),
                }
            
            return {
                "loss": outputs.loss if hasattr(outputs, 'loss') else None,
                "logits": outputs.logits if hasattr(outputs, 'logits') else None,
            }
        else:
            # Inference mode
            base_model = self.vlm.get_base_model() if hasattr(self.vlm, 'get_base_model') else self.vlm
            text_model = base_model.text_model if hasattr(base_model, 'text_model') else base_model
            
            
            text_embeds = text_model.get_input_embeddings()(input_ids)
            
            # Combine with fused features
            combined_embeds = torch.cat([fused_features, text_embeds], dim=1)
            
            # Update attention mask
            batch_size, num_img_tokens = fused_features.shape[:2]
            img_attention = torch.ones(batch_size, num_img_tokens, device=attention_mask.device if attention_mask is not None else self.device)
            if attention_mask is not None:
                combined_attention_mask = torch.cat([img_attention, attention_mask], dim=1)
            else:
                text_attention = torch.ones(batch_size, text_embeds.shape[1], device=self.device)
                combined_attention_mask = torch.cat([img_attention, text_attention], dim=1)
            
            outputs = text_model(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attention_mask,
                use_cache=False,
            )
            
            
            return {
                "logits": outputs.logits if hasattr(outputs, 'logits') else outputs,
            }
    
    def generate(
        self,
        pixel_values: Union[torch.Tensor, Dict],
        prompt: str,
        max_length: int = 128,  
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> str:
        """
        Generate text description for satellite image.
        
        Args:
            pixel_values: Satellite imagery
            prompt: Text prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling parameter
        
        Returns:
            Generated text
        """
        olmo_features = self.extract_olmoearth_features(pixel_values)
        image_embeds = self.fusion(olmo_features)  # [B, num_queries, text_embed_dim]
        
        # Use optimized generation with KV caching
        with torch.no_grad():
            # Get base model if using PEFT
            base_model = self.vlm.get_base_model() if hasattr(self.vlm, 'get_base_model') else self.vlm
            text_model = base_model.text_model if hasattr(base_model, 'text_model') else base_model
            
            # Tokenize prompt
            prompt_text = f"Question: {prompt}\nAnswer:"
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
            
            # Get text embeddings
            input_embeds = text_model.get_input_embeddings()(inputs.input_ids)
            
            # Combine image embeddings with text embeddings
            combined_embeds = torch.cat([image_embeds, input_embeds], dim=1)
            
            # Create attention mask for combined embeddings
            batch_size = combined_embeds.shape[0]
            seq_len = combined_embeds.shape[1]
            attention_mask = torch.ones(batch_size, seq_len, device=self.device)
            
            # First forward pass to get initial KV cache
            outputs = text_model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                use_cache=True,
            )
            
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Generate tokens autoregressively with KV cache
            generated_tokens = []
            
            for step in range(max_length):
                # Get next token logits (last position)
                next_token_logits = logits[:, -1, :] / max(temperature, 0.1)
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_vals = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0]
                    indices_to_remove = next_token_logits < top_k_vals[..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Check for common stop sequences
                if generated_tokens:
                    # Stop on double newline or common endings
                    decoded_so_far = self.tokenizer.decode(generated_tokens[-10:], skip_special_tokens=True)
                    if '\n\n' in decoded_so_far or '</s>' in decoded_so_far:
                        break
                
                generated_tokens.append(next_token.item())
                
                # Stop if we have enough tokens for a reasonable response
                if len(generated_tokens) >= max_length:
                    break
                
                # Get embedding for next token
                next_embed = text_model.get_input_embeddings()(next_token)
                
                # Update attention mask
                new_attention = torch.ones(batch_size, 1, device=self.device)
                attention_mask = torch.cat([attention_mask, new_attention], dim=1)
                
                # Forward pass with KV cache
                if past_key_values is not None:
                    outputs = text_model(
                        inputs_embeds=next_embed,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                else:
                    # Fallback without cache - slower
                    current_embeds = torch.cat([combined_embeds] + [text_model.get_input_embeddings()(torch.tensor([[t]], device=self.device)) for t in generated_tokens], dim=1)
                    outputs = text_model(
                        inputs_embeds=current_embeds,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            
            # Decode generated tokens
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def save_pretrained(self, save_path: str):
        """Save the model."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        torch.save(self.fusion.state_dict(), os.path.join(save_path, "fusion.pt"))
        
        self.vlm.save_pretrained(os.path.join(save_path, "vlm"))
        
        self.tokenizer.save_pretrained(os.path.join(save_path, "vlm"))
        
        config = {
            "olmoearth_dim": self.olmoearth_dim,
            "vlm_vision_dim": self.vlm_vision_dim,
        }
        import json
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):
        """Load a saved model."""
        import json
        import os
        
        # Load config
        with open(os.path.join(load_path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(**kwargs)
        
        fusion_path = os.path.join(load_path, "fusion.pt")
        if os.path.exists(fusion_path):
            try:
                fusion_state = torch.load(fusion_path, map_location=model.device)
                model.fusion.load_state_dict(fusion_state)
                logger.info("Loaded fusion weights from checkpoint")
            except RuntimeError as e:
                logger.warning(f"Could not load fusion weights (dimension mismatch): {e}")
                logger.info("Using freshly initialized fusion module - needs retraining!")
        
        # Load VLM
        from peft import PeftModel
        vlm_path = os.path.join(load_path, "vlm")
        if os.path.exists(vlm_path):
            try:
                model.vlm = PeftModel.from_pretrained(model.vlm, vlm_path)
                logger.info("Loaded VLM LoRA weights from checkpoint")
            except Exception as e:
                logger.warning(f"Could not load VLM weights: {e}")
        
        logger.info(f"Model loaded from {load_path}")
        
        return model
