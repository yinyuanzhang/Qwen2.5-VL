#!/bin/bash

# --- Configuration ---
# Set your model path and data paths here
# Example: If you downloaded the model to ~/.cache/huggingface/hub/Qwen--Qwen2.5-VL-3B-Instruct
# MODEL_PATH="/data/yinyuan/output_finetune" # Or your local model path
# MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct"
AUTO_DL_TMP="$HOME/autodl-tmp" # Or your data root directory

# POPE Dataset Paths
IMAGE_FOLDER="$AUTO_DL_TMP/playground/data/eval/pope/val2014"
QUESTION_FILE="$AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl"
ANSWERS_FILE="$AUTO_DL_TMP/playground/data/eval/pope/answers/qwen2-vl-2b-pope-origin-$(date +%Y%m%d-%H%M%S).jsonl" # Unique filename with timestamp
POPE_ANNOTATION_DIR="$AUTO_DL_TMP/playground/data/eval/pope/coco"

# YOLOv8 Model Path (Required if --use-image-segmentation is enabled)
# Ensure this path is correct and the model exists.
YOLO_MODEL_PATH="/home/zyy/LLaVA/checkpoints/yolov/yolov8n-seg.pt"

export CUDA_VISIBLE_DEVICES="6" # 指定使用 GPU 0



# --- Create Output Directory ---
mkdir -p $(dirname "$ANSWERS_FILE")

# --- Run Qwen2.5-VL Inference ---
echo "Starting Qwen2.5-VL inference on POPE dataset..."
python qwen25_vl_pope_inference.py \
    --model-path "$MODEL_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$QUESTION_FILE" \
    --answers-file "$ANSWERS_FILE" \
    --temperature 0.0 \
    --top-p 0.001 \
    --top-k 1 \
    --max-new-tokens 128 \
    --use-image-segmentation \
    --yolo-model-path "$YOLO_MODEL_PATH"

echo "Checking YOLO_MODEL_PATH: $YOLO_MODEL_PATH"
if [ ! -f "$YOLO_MODEL_PATH" ]; then
    echo "Error: YOLO model file does not exist at $YOLO_MODEL_PATH"
    exit 1
else
    echo "YOLO model file found."
fi

# --- Check if inference was successful ---
if [ ! -f "$ANSWERS_FILE" ] || [ ! -s "$ANSWERS_FILE" ]; then
    echo "Error: Inference output file not found or is empty. Cannot proceed with evaluation."
    exit 1
fi

echo "Inference completed. Results saved to $ANSWERS_FILE"
echo "Running POPE evaluation..."

# --- Call LLaVA's POPE Evaluation Script ---
# Assuming llava/eval/eval_pope.py exists in your project path relative to where you run this script
python eval_pope.py \
    --annotation-dir "$POPE_ANNOTATION_DIR" \
    --question-file "$QUESTION_FILE" \
    --result-file "$ANSWERS_FILE"

echo "POPE evaluation completed."