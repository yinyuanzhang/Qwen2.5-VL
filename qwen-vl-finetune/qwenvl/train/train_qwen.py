# qwen-vl-finetune/train_qwen.py

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import qwenvl.train.trainer
from trainer import replace_qwen2_vl_attention_class

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoConfig
)

# Import your custom processor and model
from qwenvl.utils.custom_processor import CustomQwen2_5_VLProcessor 
from qwenvl.utils.custom_qwen_generation import CustomQwen2_5_VLForConditionalGeneration

from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# =========================================================================

import torch.multiprocessing as mp # 导入 multiprocessing 模块

# try:
#     # 显式设置 expandable_segments 为 False
#     torch.cuda.memory._set_allocator_settings('expandable_segments:False')
#     print("PyTorch CUDA 分配器设置：expandable_segments 已设置为 False。")
# except Exception as e:
#     print(f"警告：无法设置 CUDA 分配器选项：{e}")

# 关键步骤：在任何 PyTorch 或 multiprocessing 相关代码之前设置启动方法
# 这必须在 DataLoader 或任何 CUDA 操作之前完成。
try:
    mp.set_start_method('spawn', force=True)
    print("多进程启动方法已设置为 'spawn'。")
except RuntimeError:
    # 如果已经设置过（例如被某个库设置），这里会捕获错误，可以忽略
    print("多进程启动方法已设置。")
# =========================================================================


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Propagate use_image_segmentation from model_args to data_args
    data_args.use_image_segmentation = model_args.use_image_segmentation
    
    # Initialize tokenizer first as it's needed for both model loading paths
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.use_image_segmentation:
        rank0_print("Loading model and processor with image segmentation support...")

        # Load original model weights into CustomQwen2_5_VLForConditionalGeneration
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

        # 直接从预训练模型加载 CustomQwen2_5_VLForConditionalGeneration
        # 假设 CustomQwen2_5_VLForConditionalGeneration 继承自 Qwen2_5_VLForConditionalGeneration
        # 并且其 __init__ 或 from_pretrained 方法能处理加载预训练权重
        model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            config=config, # 传递预加载的config可以避免重复加载
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else torch.float),
            attn_implementation=attn_implementation,
            trust_remote_code=True
        )
        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor
        data_args.model_type = "qwen2.5vl"   

    else: # Original logic for non-segmentation training
        rank0_print("Loading model and processor without image segmentation...")
        if "qwen2.5" in model_args.model_name_or_path.lower():
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            data_args.image_processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
            ).image_processor
            data_args.model_type = "qwen2.5vl"
        else:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            )
            data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
                model_args.model_name_or_path,
            )
            data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()
    
    if data_args.data_packing:
        data_module = make_supervised_data_module_packed(tokenizer=tokenizer, data_args=data_args)
    else:
        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
