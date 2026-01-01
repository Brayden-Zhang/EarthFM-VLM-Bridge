from huggingface_hub import hf_hub_download

# Download only the weights
weights_path = hf_hub_download(
    repo_id="allenai/OlmoEarth-v1-Nano",
    filename="weights.pth",
    local_dir="./olmo_nano_weights"
)

print(f"Weights downloaded to: {weights_path}")

