# File: qwenvl/utils/image_segmentation_utils.py

import os
import torch
import warnings
from ultralytics import YOLO

class ImageSegmentationHandler:
    # 不再需要 _yolo_model_instance 和 _yolo_device 作为类属性
    # 因为每个实例现在都将持有自己的模型和设备

    def __init__(self, yolo_model_path: str):
        self.yolo_model_path = yolo_model_path
        self._model_instance = None  # 实例级别的模型缓存
        self._device = None          # 实例级别的设备

        # 确保在主进程中实例化时不会立即加载模型
        # model_path 在这里仅仅被存储起来
        # print(f"ImageSegmentationHandler instance created for path: {yolo_model_path}. Model loading deferred to first call.")

    # 将 get_yolo_model 变为普通方法，访问 self. 属性
    def get_yolo_model(self):
        """
        这个方法会在每个 DataLoader worker 进程中被调用。
        它确保每个 worker 只加载一次 YOLO 模型到自己的 GPU。
        """
        if self._model_instance is None: # 检查实例级别的模型是否已加载
            # 确定当前进程应该使用的 GPU 设备
            if torch.cuda.is_available():
                local_rank_str = os.environ.get('LOCAL_RANK')
                if local_rank_str:
                    self._device = torch.device(f'cuda:{local_rank_str}')
                else:
                    self._device = torch.device('cuda:0')
                print(f"Worker loading YOLO model on GPU device: {self._device} (physical ID depends on CUDA_VISIBLE_DEVICES)")
            else:
                self._device = torch.device('cpu')
                print(f"Worker loading YOLO model on CPU: {self._device}")

            # 真正加载模型并移动到设备上
            if self.yolo_model_path: # 使用实例的 yolo_model_path
                self._model_instance = YOLO(self.yolo_model_path).eval().to(self._device)
                if not self._model_instance:
                    warnings.warn(f"Failed to load YOLO model from {self.yolo_model_path}. Segmentation will be skipped.")
            else:
                warnings.warn("YOLO model path not available in handler instance. Skipping YOLO model loading.")
                self._model_instance = None
        
        return self._model_instance

    def __call__(self, image_pil, **kwargs):
        # 在 __call__ 方法中获取已加载的模型实例
        yolo_model = self.get_yolo_model() # 调用实例方法
        if yolo_model:
            results = yolo_model(image_pil, **kwargs)
            return results
        else:
            return None