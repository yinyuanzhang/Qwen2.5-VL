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

        if self.use_image_segmentation:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoConfig
            from .custom_processor import CustomQwen2_5_VLProcessor # Make sure this imports your custom processor
            from .custom_qwen_generation import CustomQwen2_5_VLForConditionalGeneration
            
            # Initialize custom processor
            original_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            original_processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            qwen2VLImageProcessor = original_processor.image_processor
            
            self.processor = CustomQwen2_5_VLProcessor(
                image_processor=qwen2VLImageProcessor,
                tokenizer=original_tokenizer,
                chat_template=original_tokenizer.chat_template,
            )
            print(f"Loaded CustomQwen2_5_VLProcessor for {model_path}")

            # Load CUSTOM model for segmentation
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            base_model_for_weights = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                device_map='auto',
                torch_dtype='auto',
                attn_implementation='flash_attention_2', # Assuming this is always desired
                trust_remote_code=True
            )
            original_model_weights = base_model_for_weights.state_dict()
            del base_model_for_weights # Free up the temporary model
            torch.cuda.empty_cache() # Clear cache if any

            # Create an instance of your custom model
            my_custom_model = CustomQwen2_5_VLForConditionalGeneration(config)
            my_custom_model.load_state_dict(original_model_weights, strict=False)


            gpu_mems = get_gpu_memory()
            max_gpu_mem = max(gpu_mems) if gpu_mems else -1
            assert max_gpu_mem > 0, "No GPU memory detected or maximum GPU memory is not positive."

            if '72b' in self.model_path.lower() or '32b' in self.model_path.lower():
                device_map_arg = split_model()
            elif auto_split_flag():
                assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
                device_map_arg = 'auto'
            else:
                device_map_arg = 'cpu' # Fallback if no specific strategy

            # Apply device map to the custom model
            if device_map_arg == 'auto' and torch.cuda.is_available():
                my_custom_model.to('cuda')
                print(f"Custom model moved to default CUDA device for 'auto' strategy.")
            elif isinstance(device_map_arg, str) and device_map_arg in ['cpu', 'cuda']:
                my_custom_model.to(device_map_arg)
                print(f"Custom model moved to {device_map_arg}.")
            elif isinstance(device_map_arg, dict):
                 from accelerate import dispatch_model
                 my_custom_model = dispatch_model(my_custom_model, device_map=device_map_arg)
                 print(f"Custom model dispatched across devices.")
            else:
                 print(f"Warning: Unexpected device_map_arg '{device_map_arg}'. Moving model to default CUDA device.")
                 my_custom_model.to('cuda')

            self.model = my_custom_model # Assign the custom model to self.model
            self.model.eval() # Ensure custom model is in eval mode
            print(f"Loaded CustomQwen2_5_VLForConditionalGeneration for {model_path}")

            # Load YOLO model only in segmentation case
            from ultralytics import YOLO
            self.yolo_model_instance = YOLO('/data/zyy/LLaVA/checkpoints/yolov/yolov8l-seg.pt').eval()
            print("YOLO model loaded for image segmentation.")

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
                    model_path, torch_dtype='auto', device_map='cpu', attn_implementation='flash_attention_2'
                )
                self.model.cuda().eval()

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

        print(f"messages: {messages}")

        # src_image_path = os.path.join(os.environ['LMUData'], 'images', 'MMMU', '1.jpg')    
        # result = self.yolo_model(src_image_path)


        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)   # <|vision_start|><|image_pad|><|vision_end|> 视觉占位符
        images, videos = process_vision_info([messages])  # resize(PIL.Image e.g., 2072,504)
        
        # 用 YOLOv8 生成原始掩码，并将其转换为 PIL.Image 
        if self.use_image_segmentation:
                import numpy as np
                import cv2
                import matplotlib.pyplot as plt
                from PIL import Image
                
                if not images:
                    print("No images provided for segmentation. Skipping segmentation.")
                elif len(images) > 1:
                    print("Currently, only single image input is supported for segmentation. Processing the first image.")
                    pil_image = images[0]
                else:
                    pil_image = images[0]

                if pil_image and self.yolo_model_instance: # Corrected to self.yolo_model_instance
                    # Convert PIL Image to numpy array (RGB) for display and processing
                    original_image_np = np.array(pil_image)
                    original_height, original_width, _ = original_image_np.shape
                    
                    yolo_results = self.yolo_model_instance(pil_image, verbose=False) 

                    combined_mask_np = np.zeros((original_height, original_width), dtype=np.uint8)

                    if yolo_results and yolo_results[0].masks is not None and len(yolo_results[0].masks.data) > 0:
                        masks_tensor = yolo_results[0].masks.data # Tensor of shape (num_objects, H_mask, W_mask)
                        
                        # --- Efficiency Improvement for Mask Combination ---
                        masks_np_list = [m.cpu().numpy().astype(np.uint8) for m in masks_tensor]
                        
                        # Option 1: Loop and OR (generally robust)
                        for mask_np in masks_np_list:
                            resized_mask = cv2.resize(mask_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                            combined_mask_np = np.bitwise_or(combined_mask_np, resized_mask)
                        
                        print(f"Detected {len(masks_tensor)} foreground objects.")
                    else:
                        print("No foreground objects detected. Generating an all-background mask.")
                    
                    # 🚀 核心修改：将 combined_mask_np 转换为 PIL.Image.Image
                    # 重要的是将 numpy 数组转换为 PIL Image 对象，并且指定模式
                    # 对于单通道掩码，通常使用 'L' (灰度) 模式，或者转换为 'RGB' (如果 image_processor 要求)
                    # 由于 image_processor 期望类似图片的输入，我们最好将其转换为 RGB 模式
                    # 你可以将 combined_mask_np 堆叠成 3 个通道，然后转换为 PIL Image

                    num_ones = np.sum(combined_mask_np == 1)
                    num_zeros = np.sum(combined_mask_np == 0)
                    total_elements = combined_mask_np.size
                    proportion_ones = num_ones / total_elements
                    print(f"Number of foreground: {num_ones}")
                    print(f"Number of background: {num_ones}")
                    print(f"Proportion of foreground: {proportion_ones:.4f}") 
                    
                    combined_mask_rgb_np = np.stack([combined_mask_np, combined_mask_np, combined_mask_np], axis=-1)
                    combined_mask_pil = Image.fromarray(combined_mask_rgb_np, mode='RGB')
                    
                    masks = [combined_mask_pil] # 将 PIL Image 放入列表中
                    print(f"Final combined foreground mask generated with shape: {combined_mask_np.shape}")

                    if True or self.show_segment: 
                        import matplotlib.pyplot as plt 
                        
                        plt.figure(figsize=(18, 6)) # Adjust figure size as needed

                        # Subplot 1: Original Image
                        plt.subplot(1, 3, 1)
                        plt.imshow(pil_image)
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
                        blacked_out_image[combined_mask_np == 1] = [0, 0, 0] 
                        plt.imshow(blacked_out_image)
                        plt.title("Foreground Blacked Out")
                        plt.axis('off')
                        
                        plt.tight_layout()
                        plt.savefig("segmentation_vis.png") # Changed filename to be more descriptive
                        plt.close() # Close the figure to free up memory
                        print("Segmentation visualization saved to segmentation_vis.png")

                else: # This block handles cases where images is empty or YOLO model failed to load
                    print("Image list is empty or YOLO model not loaded. Skipping segmentation.")
                    masks = None # Explicitly set to None if no segmentation is performed


        if self.use_image_segmentation:
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

                is_foreground_mask = (mask_inputs.pixel_values == 1).any(dim=1) 
                self.generate_kwargs['is_foreground_mask'] = is_foreground_mask

        else:
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        

        inputs = inputs.to('cuda')
        self.model.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
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
