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


from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor



class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }

class CustomQwen2_VLProcessor(Qwen2VLProcessor):
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
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        """
        主方法，用于为模型准备一个或多个序列和图像。
        新增 masks 参数，并将原始 masks 数据包含在返回的 BatchFeature 中。
        """
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
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



        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
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
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
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
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if video_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        self.video_token, "<|placeholder|>" * (video_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}), mask_inputs