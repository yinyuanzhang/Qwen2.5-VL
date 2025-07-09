#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=1 # <--- 临时修改为1，用于单GPU调试
export CUDA_VISIBLE_DEVICES="2" # <--- 临时指定一个GPU，例如GPU 2

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
datasets=llava%1

# Output configuration
run_name="qwen2vl-baseline"
output_dir=./output

# Training arguments
# 注意：这里需要移除 --deepspeed 参数，因为单进程调试时可能不方便处理 DeepSpeed
# 如果你想在单进程下也使用 DeepSpeed 调试，需要确保 DeepSpeed 在单进程模式下也能初始化
# 但对于 'unexpected keyword argument' 这种代码逻辑错误，可以先不带 DeepSpeed
args="
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
    --save_steps 1000 \
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
    --report_to wandb"

# Launch training (临时改为直接运行 python，以便调试)
# torchrun --nproc_per_node=${NPROC_PER_NODE} \
#          --master_addr=${MASTER_ADDR} \
#          --master_port=${MASTER_PORT} \
#          ${entry_file} ${args}
python ${entry_file} ${args}