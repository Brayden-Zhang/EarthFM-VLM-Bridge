# EarthFM VLM Bridge

This codebase integrates OlmoEarth's pretrained geospatial foundation model with the Moondream2 Vision-Language Model (VLM) to enable text generation and semantic search on satellite imagery.

## Demo
Video of the Streamlit Web App working with Chat + Semantic Search functionalities of the trained VLM Model over the entire ChatEarthNet dataset:  

[![OlmoEarth VLM Bridge Demo](https://img.youtube.com/vi/fxNoTQz3XdU/maxresdefault.jpg)](https://www.youtube.com/watch?v=fxNoTQz3XdU)




## Setup

- Python 3.12
- Install the project in editable mode with dependencies:
  - `pip install -e .`
  - Optionally add dev extras you already use in this repo (for general development, `pip install -e .[dev]`).
- Make sure the ChatEarthNet data is available under `./chatearthnet_data` (or point `--data_dir` to your path when training).
    - install it by running the `tools/installing/install_chatearthnet_dataset.py`
- download OlmoEarth checkpoints using `tools/installing/download_checkpoint.py`

## Main Components

### Core Architecture

**olmoearth_vlm_bridge.py**

- **GeospatialQueryEncoder**
  - Inputs: OlmoEarth features of shape $[B, N, D_\text{olmo}]$.
  - Outputs: A set of learnable geospatial queries of shape $[B, Q, D_\text{text}]$.
  - Uses multi-head attention plus an FFN block (Perceiver-style) to let queries attend over OlmoEarth tokens.

- **OlmoEarthVLMBridge**
  - Wraps three parts:
    - OlmoEarth encoder (loaded via `ModelID`, e.g. `OLMOEARTH_V1_NANO`).
    - GeospatialQueryEncoder fusion module.
    - Moondream2 VLM text model (`vikhyatk/moondream2` by default).
  - Projects OlmoEarth latent tokens into the Moondream2 text embedding space and conditions generation on them.

Key init arguments (see code for full details):

- `olmoearth_model_id` (default: `ModelID.OLMOEARTH_V1_NANO`)
  - Controls which OlmoEarth backbone is used (NANO/TINY/BASE/LARGE).
- `vlm_model_id` (default: `"vikhyatk/moondream2"`)
  - Hugging Face model id for the VLM.
- `freeze_olmoearth` (default: `True`)
  - If `True`, OlmoEarth encoder is kept frozen; only fusion + VLM parameters train.
- `freeze_vlm_vision` (default: `False`)
  - Optionally freeze the VLM vision encoder.
- `use_lora` (default: `True`)
  - Enables LoRA adapters on key attention/MLP layers of the language model.
- `lora_r`, `lora_alpha` (defaults: `16`, `32`)
  - LoRA rank and scaling factor.
- `num_geo_queries` (default: `32`)
  - Number of geospatial query tokens used to represent the scene.
- `device` (default: `"cuda"` if available)
  - Device where the model is instantiated.

Additional behavior:

- OlmoEarth features are extracted at patch size 8 (for the nano model), reshaped to token sequences, and passed into the fusion module.
- the bridge projects OlmoEarth tokens into the text embedding space of the Phi-based language model.
- If LoRA is enabled, only LoRA and unfrozen parameters are optimized during training.

## Inference

**inference/inference_olmoearth_vlm.py**

Script to generate text descriptions for satellite images using a trained bridge model.

Main CLI arguments:

- `--model_path` (required): Path to a saved OlmoEarthVLMBridge checkpoint directory.
- `--image_path` (required): Path to the satellite RGB image.
- `--prompt`: Text prompt to steer the description (default is a generic "describe this image").
- `--max_length`, `--temperature`, `--top_p`, `--top_k`: Standard decoding parameters.
- `--device`: `cuda` or `cpu` (defaults to `cuda` if available).

Example:

```bash
python -m vlm.inference.inference_olmoearth_vlm \
  --model_path vlm_full_checkpoints/best_model \
  --image_path path/to/image.png \
  --prompt "Describe the land use and vegetation in this scene."
```

If `model_path` does not contain a `config.json`, the script will instantiate a fresh `OlmoEarthVLMBridge` with default settings.

## Training

**training/train_olmoearth_vlm.py**

Training script for the bridge model on the ChatEarthNet dataset.

Important model arguments:

- `--olmoearth_model`: One of `OLMOEARTH_V1_NANO`, `OLMOEARTH_V1_TINY`, `OLMOEARTH_V1_BASE`, `OLMOEARTH_V1_LARGE`.
- `--vlm_model`: VLM model id (default `vikhyatk/moondream2`).
- `--fusion_type`: Fusion module type (set to `geospatial_queries`)
- `--num_geo_queries`: Number of geospatial query tokens (default 32).
- `--freeze_olmoearth`: Freeze OlmoEarth encoder (on by default).
- `--freeze_vlm_vision`: Freeze the VLM vision tower.
- `--use_lora`, `--lora_r`, `--lora_alpha`: Control lora settings

Important training arguments:

- `--data_dir`: Path to ChatEarthNet data (default `./chatearthnet_data`).
- `--output_dir`: Directory for checkpoints (default `./olmoearth_vlm_checkpoints`).
- `--batch_size`, `--accumulation_steps`
- `--num_epochs`, `--learning_rate`, `--warmup_steps`, `--weight_decay`.
- `--save_every`, `--validate_every`: Steps between checkpointing and validation.
- `--device`: `cuda` or `cpu`.

Example (single GPU):

```bash
python -m vlm.training.train_olmoearth_vlm \
  --data_dir ./chatearthnet_data \
  --output_dir ./olmoearth_vlm_checkpoints \
  --olmoearth_model OLMOEARTH_V1_NANO \
  --vlm_model vikhyatk/moondream2 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --num_epochs 2 \
  --use_lora --freeze_olmoearth
```

**training/launch_olmoearth_vlm_training.sh** for shell script to launch training with a pre-defined set of hyperparameters and environment flags.


## References

Many thanks to the original authors of [OlmoEarth](https://github.com/allenai/olmoearth_pretrain), [Moondream2](https://github.com/vikhyatk/moondream2), and [ChatEarthNet](https://github.com/zhu-xlab/ChatEarthNet) for their foundational work.