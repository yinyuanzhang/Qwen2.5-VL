# qwen-vl-finetune/qwenvl/utils/image_segmentation_utils.py

import numpy as np
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import warnings

class ImageSegmentationHandler:
    def __init__(self, yolo_model_path: str):
        print(f"Loading YOLO model from {yolo_model_path} for image segmentation.")
        self.yolo_model_instance = YOLO(yolo_model_path).eval().to('cpu')
        if not self.yolo_model_instance:
            warnings.warn(f"Failed to load YOLO model from {yolo_model_path}. Segmentation will be skipped.")
        
    def generate_segmentation_mask(self, pil_image: Image.Image, show_segment: bool = False) -> Image.Image | None:
        """
        Generates a combined foreground mask from a PIL Image using YOLO.
        Returns a PIL Image representing the mask (0 for background, 1 for foreground).
        """
        if not self.yolo_model_instance:
            print("YOLO model not loaded. Skipping segmentation mask generation.")
            return None

        if not pil_image:
            print("No PIL image provided for segmentation. Skipping segmentation.")
            return None

        original_image_np = np.array(pil_image)
        original_height, original_width, _ = original_image_np.shape
        combined_mask_np = np.zeros((original_height, original_width), dtype=np.uint8)

        try:
            # YOLOv8 expects PIL Image or numpy array
            yolo_results = self.yolo_model_instance(pil_image, verbose=False)

            if yolo_results and yolo_results[0].masks is not None and len(yolo_results[0].masks.data) > 0:
                masks_tensor = yolo_results[0].masks.data # Tensor of shape (num_objects, H_mask, W_mask)
                
                for mask_np in [m.cpu().numpy().astype(np.uint8) for m in masks_tensor]:
                    # Resize mask back to original image dimensions. INTER_NEAREST is crucial for binary masks.
                    resized_mask = cv2.resize(mask_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
                    combined_mask_np = np.bitwise_or(combined_mask_np, resized_mask)
                
                print(f"Detected {len(masks_tensor)} foreground objects.")
            else:
                print("No foreground objects detected. Generating an all-background mask.")
        except Exception as e:
            warnings.warn(f"Error during YOLO segmentation: {e}. Returning an all-background mask.")
            combined_mask_np = np.zeros((original_height, original_width), dtype=np.uint8)


        # Convert combined_mask_np to a PIL Image (RGB mode)
        # We stack it to 3 channels because the image processor usually expects 3 channels.
        combined_mask_rgb_np = np.stack([combined_mask_np, combined_mask_np, combined_mask_np], axis=-1)
        combined_mask_pil = Image.fromarray(combined_mask_rgb_np, mode='RGB')
        
        print(f"Final combined foreground mask generated with shape: {combined_mask_np.shape}")

        if show_segment:
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(pil_image)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(combined_mask_np, cmap='gray')
            plt.title("Generated Foreground Mask")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            blacked_out_image = original_image_np.copy()
            blacked_out_image[combined_mask_np == 1] = [0, 0, 0] # Black out foreground
            plt.imshow(blacked_out_image)
            plt.title("Foreground Blacked Out")
            plt.axis('off')
            
            plt.tight_layout()
            # Save to a unique filename or a temporary one
            output_filename = "segmentation_vis_train.png" 
            plt.savefig(output_filename)
            plt.close()
            print(f"Segmentation visualization saved to {output_filename}")

        return combined_mask_pil

if __name__ == '__main__':
    # Example usage for testing
    # Make sure to provide a valid path to a YOLOv8 segmentation model
    # And a valid image path for testing
    yolo_model_path = '/data/zyy/LLaVA/checkpoints/yolov/yolov8l-seg.pt' # Adjust this path
    test_image_path = 'segmentation_vis.png' # Or any other test image
    
    if os.path.exists(yolo_model_path) and os.path.exists(test_image_path):
        segmentation_handler = ImageSegmentationHandler(yolo_model_path)
        test_image = Image.open(test_image_path).convert("RGB")
        mask_image = segmentation_handler.generate_segmentation_mask(test_image, show_segment=True)
        if mask_image:
            print(f"Successfully generated mask of size: {mask_image.size}")
        else:
            print("Failed to generate mask.")
    else:
        print("Please provide valid paths for YOLO model and a test image to run this example.")