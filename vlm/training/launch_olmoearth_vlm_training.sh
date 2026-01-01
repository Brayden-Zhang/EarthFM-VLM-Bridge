
echo "========================================"
echo "OlmoEarth-VLM Training Launcher"
echo "========================================"
echo ""

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

DATA_DIR="./chatearthnet_data"
OUTPUT_DIR="./olmoearth_vlm_checkpoints"
BATCH_SIZE=4
ACCUM_STEPS=4
NUM_EPOCHS=10
LR=2e-5
SAVE_EVERY=500
VALIDATE_EVERY=200

echo "Training Configuration:"
echo "  Data: $DATA_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $ACCUM_STEPS"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LR"
echo ""

mkdir -p "$OUTPUT_DIR"

echo "Starting training..."
echo ""

python train_olmoearth_vlm.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size $BATCH_SIZE \
  --accumulation_steps $ACCUM_STEPS \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LR \
  --save_every $SAVE_EVERY \
  --validate_every $VALIDATE_EVERY \
  --fusion_type geospatial_queries \
  --num_geo_queries 32 \
  --freeze_olmoearth \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32

echo ""
echo "========================================"
echo "Training completed!"
echo "========================================"
