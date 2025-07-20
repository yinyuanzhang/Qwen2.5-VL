        if self.use_image_segmentation:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoConfig
            from .qwen25vl.custom_processor import CustomQwen2_5_VLProcessor # Make sure this imports your custom processor
            from .qwen25vl.custom_qwen_generation import CustomQwen2_5_VLForConditionalGeneration
            
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


            # Load original model weights into CustomQwen2_5_VLForConditionalGeneration
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

            # 直接从预训练模型加载 CustomQwen2_5_VLForConditionalGeneration
            # 假设 CustomQwen2_5_VLForConditionalGeneration 继承自 Qwen2_5_VLForConditionalGeneration
            # 并且其 __init__ 或 from_pretrained 方法能处理加载预训练权重
            my_custom_model = CustomQwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                config=config, # 传递预加载的config可以避免重复加载
                attn_implementation='flash_attention_2',
                trust_remote_code=True
            )
            print(f"Loaded CustomQwen2_5_VLProcessor for {model_path}")

            gpu_mems = get_gpu_memory()
            max_gpu_mem = max(gpu_mems) if gpu_mems else -1
            assert max_gpu_mem > 0, "No GPU memory detected or maximum GPU memory is not positive."

            if '72b' in self.model_path.lower() or '32b' in self.model_path.lower():
                device_map_arg = split_model()
            elif auto_split_flag():
                assert world_size == 1, 'Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL'
                device_map_arg = 'auto'
            else:
                device_map_arg = 'cuda' # Fallback if no specific strategy

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

            self.model = my_custom_model.to(torch.bfloat16) # Assign the custom model to self.model
            self.model.eval() # Ensure custom model is in eval mode
            print(f"Loaded CustomQwen2_5_VLForConditionalGeneration for {model_path}")

            # Load YOLO model only in segmentation case
            from ultralytics import YOLO
            self.yolo_model_instance = YOLO('/home/zyy/LLaVA/checkpoints/yolov/yolov8n-seg.pt').eval()
            print("YOLO model loaded for image segmentation.")
