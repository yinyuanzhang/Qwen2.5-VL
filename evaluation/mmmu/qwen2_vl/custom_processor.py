# custom_processor.py
# 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
# 注意：此文件仅根据您的两个核心要求进行修改：
# 1. __call__ 方法新增 `masks` 参数。
# 2. `masks` 数据（未经处理）包含在最终返回的 BatchFeature 中。
# 其他原始逻辑保持不变。
# 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨

from typing import List, Union, Optional
# import torch
import numpy as np
# import cv2

# 从 transformers 库导入必要的基类和原始处理器
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor as OriginalQwen2_5_VLProcessor


# 保持原始的 Kwargs 定义不变
class Qwen2_5_VLVideosProcessorKwargs(VideosKwargs, total=False):
    fps: Union[List[float], float]

class Qwen2_5_VLProcessorKwargs(ProcessingKwargs, total=False):
    videos_kwargs: Qwen2_5_VLVideosProcessorKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "videos_kwargs": {"fps": 2.0},
    }

class CustomQwen2_5_VLProcessor(OriginalQwen2_5_VLProcessor):
    """
    自定义的 Qwen2.5-VL 处理器，继承自原始处理器，并仅在 __call__ 方法中添加 masks 参数，
    并将原始 masks 数据包含在返回的 BatchFeature 中。
    可以说，唯一的作用就是将masks与image进行了相同的patch处理。
    """
    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        # 你的 __init__ 逻辑保持不变
        super().__init__(image_processor, tokenizer, chat_template=chat_template)


    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        masks: ImageInput = None,# 新增的 masks 参数
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """
        主方法，用于为模型准备一个或多个序列和图像。
        新增 masks 参数，并将原始 masks 数据包含在返回的 BatchFeature 中。
        """
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            mask_processing_kwargs = output_kwargs["images_kwargs"].copy() # Get a copy of original kwargs
            mask_processing_kwargs["do_normalize"] = False # Disable normalization for masks
            mask_processing_kwargs["do_rescale"] = False
            
            mask_inputs = self.image_processor(images=masks, videos=None, **mask_processing_kwargs)

            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None

        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["images_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]

            fps = output_kwargs["videos_kwargs"].pop("fps", 2.0)
            if isinstance(fps, (int, float)):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / fps] * len(video_grid_thw)
            elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
                second_per_grid_ts = [self.image_processor.temporal_patch_size / tmp for tmp in fps]
            else:
                raise ValueError(
                    f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number."
                )
            videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    if index >= len(image_grid_thw):
                        raise IndexError(f"Not enough image_grid_thw entries for image token {index}. Check your input images.")
                    text[i] = text[i].replace(
                        self.image_token,
                        "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    if index >= len(video_grid_thw):
                        raise IndexError(f"Not enough video_grid_thw entries for video token {index}. Check your input videos.")
                    text[i] = text[i].replace(
                        self.video_token,
                        "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length),
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # 构建最终的 BatchFeature
        # 关键修改：将 masks 也加入到 BatchFeature 中
        final_data = {**text_inputs, **image_inputs, **videos_inputs}


        return BatchFeature(data=final_data), mask_inputs
