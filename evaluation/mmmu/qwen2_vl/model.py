from __future__ import annotations

import os
import sys
import warnings
import math
import logging

import torch

from .base import BaseModel
from .prompt import Qwen2VLPromptMixin
from .util import get_rank_and_world_size, get_gpu_memory, auto_split_flag, listinstr

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Optional


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        post_process: bool = False,  # if True, will try to only extract stuff in the last \boxed{}.
        verbose: bool = False,
        use_image_segmentation: bool = False, 
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.post_process = post_process
        self.fps = 2.0
        self.nframe = 64
        self.FRAME_FACTOR = 2
        rank, world_size = get_rank_and_world_size()
        assert model_path is not None
        self.model_path = model_path
        self.use_image_segmentation = use_image_segmentation

        MODEL_CLS = None



        from transformers import (
            AutoConfig,
            AutoProcessor, # Global import
            AutoTokenizer,
            Qwen2VLProcessor,                   # Global import
        )

        is_qwen2_5_vl = listinstr(['2.5', '2_5', 'qwen25'], model_path.lower())

        if self.use_image_segmentation:
            # --- 自定义分割模型逻辑 ---

            if is_qwen2_5_vl:
                from .qwen25vl.custom_processor import CustomQwen2_5_VLProcessor as CUSTOM_PROCESSOR_CLS
                from .qwen25vl.custom_qwen_generation import CustomQwen2_5_VLForConditionalGeneration as CUSTOM_MODEL_CLS
                BASE_PROCESSOR_CLS =  AutoProcessor
                yolo_model_path = '/home/zyy/LLaVA/checkpoints/yolov/yolov8n-seg.pt'
                print("Configuration set for Custom Qwen2.5-VL.")
            else:
                from .qwen2vl.custom_processor import CustomQwen2_VLProcessor as CUSTOM_PROCESSOR_CLS
                from .qwen2vl.custom_qwen_generation import CustomQwen2_VLForConditionalGeneration as CUSTOM_MODEL_CLS
                BASE_PROCESSOR_CLS =  Qwen2VLProcessor
                yolo_model_path = '/home/zyy/LLaVA/checkpoints/yolov/yolov8n-seg.pt'
                print("Configuration set for Custom Qwen2-VL.")

            # 统一的处理器初始化
            original_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            base_processor = BASE_PROCESSOR_CLS.from_pretrained(model_path, trust_remote_code=True)
            
            self.processor = CUSTOM_PROCESSOR_CLS(
                image_processor=base_processor.image_processor,
                tokenizer=original_tokenizer,
                chat_template=original_tokenizer.chat_template,
            )
            print(f"Loaded {self.processor.__class__.__name__} for {model_path}")

            # 统一的模型加载
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            my_custom_model = CUSTOM_MODEL_CLS.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                config=config,
                attn_implementation='flash_attention_2',
                trust_remote_code=True
            )
            print(f"Instantiated {my_custom_model.__class__.__name__} from pretrained weights.")

            # 统一的设备映射逻辑
            gpu_mems = get_gpu_memory()
            max_gpu_mem = max(gpu_mems) if gpu_mems else -1
            assert max_gpu_mem > 0, "No GPU memory detected or it's not positive."

            if '72b' in self.model_path.lower() or '32b' in self.model_path.lower():
                device_map_arg = split_model()
            elif auto_split_flag():
                assert world_size == 1, 'AUTO_SPLIT only supports world_size=1 for this model size.'
                device_map_arg = 'auto'
            else:
                device_map_arg = 'cuda'

            # 应用设备映射
            if isinstance(device_map_arg, dict):
                from accelerate import dispatch_model
                my_custom_model = dispatch_model(my_custom_model, device_map=device_map_arg)
                print("Custom model dispatched across multiple devices.")
            else:
                my_custom_model.to(device_map_arg)
                print(f"Custom model moved to '{device_map_arg}'.")

            self.model = my_custom_model # .to(torch.bfloat16) 已在from_pretrained中处理
            self.model.eval()
            print(f"Loaded and prepared custom model for {model_path}")

            # 加载 YOLO 模型
            from ultralytics import YOLO
            self.yolo_model_instance = YOLO(yolo_model_path).eval()
            print(f"YOLO model loaded from {yolo_model_path} for image segmentation.")




        else:
            if listinstr(['2.5', '2_5', 'qwen25'], model_path.lower()):
                from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
                MODEL_CLS = Qwen2_5_VLForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(model_path)
            else:
                from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
                MODEL_CLS = Qwen2VLForConditionalGeneration
                self.processor = Qwen2VLProcessor.from_pretrained(model_path)

            gpu_mems = get_gpu_memory()
            max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
            assert max_gpu_mem > 0

            # If only one process and GPU memory is less than 40GB
            if '72b' in self.model_path.lower() or '32b' in self.model_path.lower():
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map=split_model(), attn_implementation='flash_attention_2'
                )
                self.model.eval()
            elif auto_split_flag():
                assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
                # Will Use All GPUs to run one model
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
                )
            else:
                self.model = MODEL_CLS.from_pretrained(
                    model_path, torch_dtype='auto', device_map='cuda:0', attn_implementation='flash_attention_2'
                )
                self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def _plot_segmentation_results(self, original_image_pil, combined_mask_np, save_path="segmentation_vis.png"):
        """
        封装绘图逻辑，用于可视化分割结果。
        """
        if not True:
            return

        original_image_np = np.array(original_image_pil)
        
        plt.figure(figsize=(18, 6))

        # Subplot 1: Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image_pil)
        plt.title("Original Image")
        plt.axis('off')

        # Subplot 2: Foreground Mask
        plt.subplot(1, 3, 2)
        plt.imshow(combined_mask_np, cmap='gray')
        plt.title("Generated Foreground Mask")
        plt.axis('off')

        # Subplot 3: Combined Image (Background original, Foreground blacked out)
        plt.subplot(1, 3, 3)
        blacked_out_image = original_image_np.copy()
        # 注意：如果掩码是 0/1，1代表前景，则将其设置为黑色
        blacked_out_image[combined_mask_np == 1] = [0, 0, 0]
        plt.imshow(blacked_out_image)
        plt.title("Foreground Blacked Out")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() # Close the figure to free up memory
        print(f"Segmentation visualization saved to {save_path}")

    def _generate_and_process_mask(self, pil_image: Image.Image) -> Optional[List[Image.Image]]:
            """
            使用 YOLOv8 生成掩码，合并并转换为 PIL.Image 格式。
            返回一个包含合并后掩码 PIL Image 的列表。
            """
        
            original_width, original_height = pil_image.size # PIL Image 的 size 是 (width, height)
            
            yolo_results = self.yolo_model_instance(pil_image, verbose=True)

            combined_mask_np = np.zeros((original_height, original_width), dtype=np.uint8)

            if yolo_results and yolo_results[0].masks is not None and len(yolo_results[0].masks.data) > 0:
                masks_tensor = yolo_results[0].masks.data # Tensor of shape (num_objects, H_mask, W_mask)
                
                for mask_tensor in masks_tensor:
                    # 将 PyTorch Tensor 移到 CPU 并转换为 NumPy 数组
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    # 将掩码缩放到原始图像尺寸，使用 INTER_NEAREST 保持二值性
                    resized_mask = cv2.resize(mask_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                    combined_mask_np = np.bitwise_or(combined_mask_np, resized_mask)
                
            # 统计掩码中的前景和背景像素数量
            num_foreground_pixels = np.sum(combined_mask_np == 1)
            num_background_pixels = np.sum(combined_mask_np == 0)
            total_pixels = combined_mask_np.size
            
            print(f"Combined Mask Stats:")
            print(f"  Foreground Pixels: {num_foreground_pixels}")
            print(f"  Background Pixels: {num_background_pixels}")
            print(f"  Proportion of Foreground: {num_foreground_pixels / total_pixels:.4f}") 
            
            # 将合并后的单通道 NumPy 掩码转换为 PIL.Image 对象
            # 堆叠成 3 个通道（RGB）以符合 image_processor 可能的期望
            # 如果 image_processor 接受 'L' 模式（灰度）的 PIL Image，则可以简化为 Image.fromarray(combined_mask_np, mode='L')
            combined_mask_rgb_np = np.stack([combined_mask_np, combined_mask_np, combined_mask_np], axis=-1)
            combined_mask_pil = Image.fromarray(combined_mask_rgb_np, mode='RGB')
            
            print(f"Final combined foreground mask generated with shape: {combined_mask_np.shape}")

            # 可视化结果
            # self._plot_segmentation_results(pil_image, combined_mask_np)

            return [combined_mask_pil] # 总是返回一个列表，即使只有一个掩码    

    def generate_inner(self, message, dataset=None):
        try:
            from qwen_vl_utils import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')


        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)   # <|vision_start|><|image_pad|><|vision_end|> 视觉占位符
        images, videos = process_vision_info([messages])  # resize(PIL.Image e.g., 2072,504)
        

        # 图像分割逻辑
        if self.use_image_segmentation and images: # 只有当开启分割且有图像时才进行
            if (len(images) > 1):
                print("Too many images")
                        
            first_image = images[0] # YOLOv8 每次处理一张图片，所以取第一张   也不是，如果多张图片，有问题了。
            masks = self._generate_and_process_mask(first_image)
        elif self.use_image_segmentation and not images:
            print("Image segmentation enabled but no images found. Skipping segmentation.")



        if self.use_image_segmentation and masks:
            inputs, mask_inputs = self.processor(text=text, images=images, videos=videos, masks = masks, padding=True, return_tensors='pt')
            if mask_inputs and 'pixel_values' in mask_inputs:
                processed_row_mask_to_pass = (mask_inputs.pixel_values.any(dim=1)).int()

                num_ones = torch.sum(processed_row_mask_to_pass).item()
                num_zeros = len(processed_row_mask_to_pass) - num_ones
                total_elements = len(processed_row_mask_to_pass)
                proportion_ones = num_ones / total_elements
                print(f"Number of foreground in processed_row_mask: {num_ones}")
                print(f"Number of background in processed_row_mask: {num_zeros}")
                print(f"Proportion of foreground: {proportion_ones:.4f}")

                is_foreground_mask = (mask_inputs.pixel_values == 1).any(dim=1)   # [5328,1176]
                self.generate_kwargs['is_foreground_mask'] = is_foreground_mask

        else:
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        

        inputs = inputs.to('cuda')
        # self.model.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            do_sample=False,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = out[0]
        
        if self.post_process:
            resp = response.split('\\boxed{')[-1]
            lt = len(resp)
            counter, end = 1, None
            for i in range(lt):
                if resp[i] == '{':
                    counter += 1
                elif resp[i] == '}':
                    counter -= 1
                if counter == 0:
                    end = i
                    break
                elif i == lt - 1:
                    end = lt
                    break
            if end is not None:
                response = resp[:end]

        if self.verbose:
            print(f'\033[32m{response}\033[0m')
        return response
