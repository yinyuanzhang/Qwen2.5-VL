#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
# NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l) # 自动检测可用的GPU数量
NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/data/zyy/cuda-11.8/libcurand/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False


# locate libmpi_cxx
export NCCL_DEBUG=INFO
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=llava

# Output configuration
run_name="qwen2vl-baseline"
output_dir=/data/yinyuan/output_finetune

# Image segmentation configuration (new parameters)
USE_IMAGE_SEGMENTATION=True # Set to True to enable image segmentation
YOLO_MODEL_PATH=/home/zyy/LLaVA/checkpoints/yolov/yolov8n-seg.pt # Path to your YOLOv8 segmentation model

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb \
    --use_image_segmentation ${USE_IMAGE_SEGMENTATION} \
    --yolo_model_path "${YOLO_MODEL_PATH}"
"


# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}