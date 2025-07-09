import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
from torchcodec.decoders import VideoDecoder
import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

# ====================================================================
import warnings
import cv2
from qwenvl.utils.image_segmentation_utils import ImageSegmentationHandler
from qwenvl.utils.custom_processor import CustomQwen2_5_VLProcessor
# ====================================================================

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        random.shuffle(list_data_dict)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.image_processor.size["shortest_edge"] = data_args.min_pixels

        # ====================================================================
        # 新增：初始化 ImageSegmentationHandler
        self.use_image_segmentation = getattr(data_args, "use_image_segmentation", False)
        self.yolo_model_path = getattr(data_args, "yolo_model_path", None)
        self.yolo_model_instance = None
        if self.use_image_segmentation and self.yolo_model_path:
            try:
                # 假设 ImageSegmentationHandler 已经被正确导入
                from qwenvl.utils.image_segmentation_utils import ImageSegmentationHandler
                self.yolo_segmentation_handler = ImageSegmentationHandler(self.yolo_model_path)
                self.yolo_model_instance = self.yolo_segmentation_handler.yolo_model_instance
            except Exception as e:
                warnings.warn(f"Failed to initialize ImageSegmentationHandler: {e}. Image segmentation will be disabled.")
                self.use_image_segmentation = False
        # ====================================================================


    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")

        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw, image

    def process_video(self, video_file):
        decord_video = None
        decord_attempts = 0
        max_decord_attempts = 3
        while decord_attempts < max_decord_attempts:
            try:
                decord_video = self.video_decord(video_file)
                return decord_video
                if decord_video:
                    break
            except Exception as e:
                print(f"Decord attempt {decord_attempts + 1} failed: {e}")
                decord_attempts += 1

        torchcodec_video = None
        try:
            torchcodec_video = self.video_torchcodec(video_file)
            return torchcodec_video
        except Exception as e:
            print(f"torchcodec attempt failed: {e}")

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        device = "cpu"  # or e.g. "cuda"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            images=None, videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.image_processor.temporal_patch_size / fps
        ] * len(grid_thw)
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                # ====================================================================
                # 如果 _get_item 返回 None，我们在此处重试，而不是直接返回 None 导致 collator 崩溃
                if sample is not None:
                    return sample
                else:
                    # 如果 _get_item 明确返回 None，表明该样本有问题，尝试下一次
                    print(f"[Try #{attempt_idx}] _get_item returned None for sample {i}.")
                    time.sleep(1) # 仍然等待，可能是临时文件问题
                # ====================================================================
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)


        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries): # Changed to num_final_retries for more robustness
            try:
                next_index = random.choice(range(len(self))) # 随机选择其他样本，而不是仅仅是 i+1
                # next_index = min(i + 1, len(self.list_data_dict) - 1) # Keep this if you prefer sequential retry
                sample = self._get_item(next_index)
                # ====================================================================
                # 再次检查 sample 是否为 None
                if sample is not None:
                    return sample
                else:
                    print(f"[Try other #{attempt_idx}] _get_item returned None for sample {next_index}.")
                # ====================================================================
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i) # Final attempt on the original index
            # ====================================================================
            if sample is None: # 如果最终还是 None，就抛出错误
                raise ValueError(f"Failed to fetch sample {i} after all retries; _get_item consistently returned None.")
            # ====================================================================
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # define some variables
        grid_thw_merged = None
        video_grid_thw_merged = None
        grid_thw = None
        video_grid_thw = None
        second_per_grid_ts = None

        # ====================================================================
        pil_images_raw = [] # 用于存储原始 PIL 图像，以便进行分割
        # ====================================================================
        pil_foreground_masks = [] # 用于存储生成的 PIL 掩码

        if "image" in sources[0]:
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file_paths = [
                        os.path.join(image_folder, file) for file in image_file
                    ]
                    # =========================================================
                    # 修改：process_image_unified 返回原始 PIL 图像
                    results = [self.process_image_unified(file) for file in image_file_paths]
                    image_tensors, grid_thws, original_pil_images = zip(*results)
                    image = list(image_tensors)
                    grid_thw = list(grid_thws)
                    pil_images_raw.extend(original_pil_images) # 收集原始 PIL 图像
                    # =========================================================
                else:
                    image_file_path = image_file[0]
                    image_file_path = os.path.join(image_folder, image_file_path)
                    # =========================================================
                    # 修改：process_image_unified 返回原始 PIL 图像
                    image_tensor, grid_thw_single, original_pil_image_single = self.process_image_unified(image_file_path)
                    image = [image_tensor]
                    grid_thw = [grid_thw_single]
                    pil_images_raw.append(original_pil_image_single) # 收集原始 PIL 图像
                    # =========================================================
            else:
                image_file_path = os.path.join(image_folder, image_file)
                # =========================================================
                # 修改：process_image_unified 返回原始 PIL 图像
                image_tensor, grid_thw_single, original_pil_image_single = self.process_image_unified(image_file_path)
                image = [image_tensor]
                grid_thw = [grid_thw_single]
                pil_images_raw.append(original_pil_image_single) # 收集原始 PIL 图像
                # =========================================================
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in grid_thw_merged
            ]

            # ====================================================================
            # 新增：执行图像分割并生成掩码
            if self.use_image_segmentation and self.yolo_model_instance and pil_images_raw:
                try:
                    # 你的推理代码片段，现在适应 _get_item 的上下文
                    # 遍历每一个原始 PIL 图像进行分割
                    for idx, pil_image in enumerate(pil_images_raw):
                        print(f"===== Processing image {idx+1}/{len(pil_images_raw)} for segmentation =====")
                        
                        original_image_np = np.array(pil_image)
                        original_height, original_width, _ = original_image_np.shape
                        
                        yolo_results = self.yolo_model_instance(pil_image, verbose=False) 

                        combined_mask_np = np.zeros((original_height, original_width), dtype=np.uint8)

                        if yolo_results and len(yolo_results) > 0 and yolo_results[0].masks is not None and len(yolo_results[0].masks.data) > 0:
                            masks_tensor = yolo_results[0].masks.data # Tensor of shape (num_objects, H_mask, W_mask)
                            
                            masks_np_list = [m.cpu().numpy().astype(np.uint8) for m in masks_tensor]
                            
                            for mask_np in masks_np_list:
                                # 确保 resize 到原始图像尺寸
                                resized_mask = cv2.resize(mask_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                                combined_mask_np = np.bitwise_or(combined_mask_np, resized_mask)
                            
                            print(f"Detected {len(masks_tensor)} foreground objects for image {idx+1}.")
                        else:
                            print(f"No foreground objects detected for image {idx+1}. Generating an all-background mask.")
                        
                        # 转换 combined_mask_np 为 PIL.Image.Image
                        # 由于 CustomQwen2_5_VLProcessor 期望 PIL.Image，这里转换
                        # 并且为了与 image_processor 兼容，通常将单通道 mask 转换为 RGB 
                        combined_mask_rgb_np = np.stack([combined_mask_np, combined_mask_np, combined_mask_np], axis=-1)
                        combined_mask_pil = Image.fromarray(combined_mask_rgb_np, mode='RGB')
                        
                        pil_foreground_masks.append(combined_mask_pil)

                    # 过滤掉可能因为某些原因生成 None 的掩码（尽管上面的逻辑尽量避免）
                    pil_foreground_masks = [mask for mask in pil_foreground_masks if mask is not None]
                    if not pil_foreground_masks:
                        # 如果所有掩码都失败了，则设置 masks 为 None，或返回 None 样本
                        logging.warning(f"All foreground masks failed for sample {i}. This sample might be invalid.")
                        # 此时可以选择返回 None，让外层 __getitem__ 重试其他样本
                        # 或者传递空的 mask 列表给 CustomQwen2_5_VLProcessor，让其内部处理
                        # 为了避免 NoneType 错误，如果这里没有有效的掩码，我们最好返回 None 让整个样本重试
                        return None 
                except Exception as e:
                    warnings.warn(f"Error during YOLO segmentation for sample {i}: {e}. Skipping segmentation for this sample.")
                    pil_foreground_masks = [] # 清空，确保不传入错误的掩码
                    return None # 返回 None，让外层 __getitem__ 重试其他样本
            else:
                # 如果不使用分割，或者 YOLO 模型未加载，或者没有图像，则不生成掩码
                pil_foreground_masks = []
            # ====================================================================

        if "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [
                        os.path.join(video_folder, file) for file in video_file
                    ]
                    results = [self.process_video(file) for file in video_file]
                    video, video_grid_thw, second_per_grid_ts = zip(*results)
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, video_grid_thw, second_per_grid_ts = self.process_video(
                        video_file
                    )
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, video_grid_thw, second_per_grid_ts = self.process_video(
                    video_file
                )
                video = [video]
            video_grid_thw_merged = copy.deepcopy(video_grid_thw)
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw_merged = [video_grid_thw_merged]
                video_grid_thw = [video_grid_thw]
            video_grid_thw_merged = [
                merged_thw.prod() // self.data_args.image_processor.merge_size**2
                for merged_thw in video_grid_thw_merged
            ]
        chat_sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess_qwen_2_visual(
            chat_sources,
            self.tokenizer,
            grid_thw_image=grid_thw_merged if grid_thw_merged else None,
            grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
        )
        position_ids, _ = self.get_rope_index(
            self.data_args.image_processor.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )
        if "image" not in sources[0] and "video" not in sources[0]:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(
                sources, self.tokenizer, grid_thw=grid_thw_merged
            )
            position_ids = (
                torch.arange(0, data_dict["input_ids"].size(1))
                .view(1, -1)
                .unsqueeze(0)
                .expand(3, -1, -1)
            )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

        # ====================================================================
        # 使用 CustomQwen2_5_VLProcessor 替换原始 processor
        # 确保 CustomQwen2_5_VLProcessor 已经能够处理 masks 参数
        # 这里需要替换 self.data_args.image_processor 为 CustomQwen2_5_VLProcessor 实例
        # 考虑到 data_args.image_processor 已经是 Qwen2VLImageProcessor，
        # 我们需要在 __init__ 中或这里创建一个 CustomQwen2_5_VLProcessor 实例
        
        # 最佳实践：在 LazySupervisedDataset 的 __init__ 中替换掉 image_processor
        # 确保 self.data_args.image_processor 实际上是一个 CustomQwen2_5_VLProcessor 实例
        # 并且你的 CustomQwen2_5_VLProcessor 已经能处理 images 和 masks
        # 因此，_get_item 中只使用 self.data_args.image_processor.preprocess 即可
        
        # 之前你的代码直接调用了 CustomQwen2_5_VLProcessor 的 __call__ 方法
        # `inputs, mask_inputs = self.processor(text=text, images=images, videos=videos, masks = masks, padding=True, return_tensors='pt')`
        # 在数据集中，我们不直接调用整个 processor，而是调用其内部的 image_processor 部分
        # 因此，下面的部分是模拟 CustomQwen2_5_VLProcessor 内部对图像和掩码的处理逻辑
        
        # 如果 self.data_args.image_processor 已经被替换为 CustomQwen2_5_VLProcessor
        # 那么它会自己处理 masks 参数。
        # 这里我们收集 pil_images_raw 和 pil_foreground_masks，然后将其放入 data_dict
        # 稍后 collator 会将它们打包。
        # ====================================================================


        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = torch.cat(image, dim=0)
            data_dict["image_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in grid_thw], dim=0
            )
            # ====================================================================
            # 新增：添加原始 PIL 图像和生成的 PIL 掩码到 data_dict
            # Collator 将需要这些原始 PIL 对象来再次调用 image_processor 进行最终的张量化
            data_dict["pil_images"] = pil_images_raw
            # 过滤掉 None 掩码以防万一
            data_dict["pil_foreground_masks"] = [mask for mask in pil_foreground_masks if mask is not None]
            # ====================================================================
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
            data_dict["video_grid_thw"] = torch.cat(
                [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
            )

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        
        # ====================================================================
        # 从实例中收集原始 PIL 图像和生成的 PIL 掩码
        # 这些是 CustomQwen2_5_VLProcessor 所需的输入
        pil_images = []
        pil_foreground_masks = []
        
        # 确保只在图像存在且使用了图像分割时处理
        # 并且需要检查实例中是否有 'pil_images' 和 'pil_foreground_masks'
        # 这里的逻辑将适用于 DataCollatorForSupervisedDataset 和 FlattenedDataCollatorForSupervisedDataset
        for instance in instances:
            if "pil_images" in instance and instance["pil_images"]:
                pil_images.extend(instance["pil_images"])
            # 过滤掉任何可能为 None 的掩码
            if "pil_foreground_masks" in instance and instance["pil_foreground_masks"]:
                pil_foreground_masks.extend([mask for mask in instance["pil_foreground_masks"] if mask is not None])
        # ====================================================================


        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )

        # ====================================================================
        # 新增：如果使用图像分割，则使用 CustomQwen2_5_VLProcessor 处理图像和掩码
        # 假设 self.tokenizer (作为 processor) 已经是一个 CustomQwen2_5_VLProcessor 的实例
        # 并且 CustomQwen2_5_VLProcessor 的 __call__ 方法支持 images 和 masks 参数
        # 并且在 CustomQwen2_5_VLProcessor 的 __init__ 中会处理 data_args.image_processor
        # 所以这里的 self.tokenizer 实际上是 CustomQwen2_5_VLProcessor 实例
        
        # NOTE: 这里的逻辑需要根据 train_qwen.py 中 Trainer 的 processor 参数如何传递来调整
        # 如果 Trainer(processing_class=tokenizer, ...) 这里的 tokenizer 是 CustomQwen2_5_VLProcessor
        # 那么在这里调用它，并传入 images 和 masks
        # 否则，你需要访问模型实际的 image_processor
        
        # 考虑到你的 CustomQwen2_5_VLProcessor 设计和目标，它应该是一个完整的处理器，包含 image_processor 和 tokenizer
        # 所以在这里，我们应该使用传入的 CustomQwen2_5_VLProcessor 实例（即 self.tokenizer）来处理图像和掩码
        
        # 检查是否应该使用 image segmentation
        # 从 data_args 传递 use_image_segmentation 到 collator
        # 为了简化，假设 DataCollatorForSupervisedDataset 实例可以访问到这个信息
        # 通常这通过将其作为 collator 的一个属性来完成
        
        # NOTE: 在 make_supervised_data_module 中，需要将 data_args 传递给 collator
        # data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
        # 并在 __init__ 中保存 self.data_args = data_args
        # 暂时先这样处理，如果报错再回头修改 collator 的 __init__
        
        # 假设 self.use_image_segmentation 属性在 collator 中可用
        # 但在当前代码中，collator 只有 tokenizer。这需要额外修改。
        # 为了不修改太多，我们假设 processed_visual_info 已经包含了 masks 的 pixel_values
        
        # 这段逻辑应在 LazySupervisedDataset 的 _get_item 中完成图像和掩码的最终处理和张量化
        # Collator 应该只负责打包。
        # 让我们重新考虑一下。_get_item 应该返回所有处理好的张量。
        # Collator 只是对这些张量进行 padding 和 stacking。
        
        # 如果 _get_item 已经将 masks 处理成 tensor 并放入 data_dict['is_foreground_mask']
        # 那么 collator 只需要收集它。
        # ====================================================================


        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        # ====================================================================
        # 新增：收集 is_foreground_mask
        is_foreground_masks = list(
            instance["is_foreground_mask"]
            for instance in instances
            if "is_foreground_mask" in instance and instance["is_foreground_mask"] is not None
        )
        if len(is_foreground_masks) != 0:
            # 确保这些掩码可以被堆叠。如果它们已经是 pixel_values 格式，直接堆叠
            # 如果是布尔掩码，也确保堆叠。
            # 这里假设它们是已经处理好的张量
            concat_foreground_masks = torch.cat([mask for mask in is_foreground_masks], dim=0)
        else:
            concat_foreground_masks = None
        # ====================================================================

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        # ====================================================================
        batch["is_foreground_mask"] = concat_foreground_masks
        # ====================================================================

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    if data_args.data_flatten:
        # ====================================================================
        # 传入 data_args 给 Collator 以便其判断是否使用图像分割
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
        # NOTE: data_collator 必须能够访问 self.use_image_segmentation
        # 最简单的方法是在 DataCollatorForSupervisedDataset 的 __init__ 中接受并存储 data_args
        # 或者直接通过 CustomQwen2_5_VLProcessor 传递相关信息
        # 考虑到 Trainer 的处理方式，collator 最终会接收由 _get_item 返回的字典
        # 因此，collator 只需要负责打包 _get_item 已经处理好的数据
        # 所以 collator 不需要直接知道 use_image_segmentation
        # ====================================================================
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )