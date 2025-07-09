import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
import torch
import warnings
import shortuuid
import traceback
import math # Added for split_list/get_chunk, even if not fully used

# Assume these are available from your Qwen2.5-VL repository setup
try:
    from qwen2_vl.model import Qwen2VLChat
    from qwen_vl_utils import process_vision_info
    from dataset_utils import dump_image # This dump_image is the one that needs to be called
except ImportError as e:
    print(f"Error importing Qwen2.5-VL specific modules: {e}")
    print("Please ensure 'qwen2_vl.model', 'qwen_vl_utils', and 'dataset_utils' are correctly configured in your PYTHONPATH or located in the same directory.")
    sys.exit(1)

def run_qwen_pope_inference(args):
    """
    Run inference on the POPE dataset using Qwen2.5-VL,
    producing output compatible with LLaVA's eval_pope.py.
    """
    print(f"Loading questions from {args.question_file}")
    questions = []
    with open(os.path.expanduser(args.question_file), 'r') as f:
        for line in f:
            questions.append(json.loads(line))

    # Splitting logic (as in LLaVA, potentially for distributed eval)
    def split_list(lst, n):
        chunk_size = math.ceil(len(lst) / n)
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    def get_chunk(lst, n, k):
        chunks = split_list(lst, n)
        return chunks[k]

    # Apply chunking if specified (otherwise, num_chunks=1, chunk_idx=0 means all data)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    print(f"Total questions to process (after chunking): {len(questions)}")

    img_root = os.path.expanduser(args.image_folder)
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)

    # --- THE CRITICAL FIX ---
    # Redefine qwen_dump_image_wrapper to only accept 'line_dict_from_model'
    # and pass 'img_root' directly.
    def qwen_dump_image_wrapper(line_dict_from_model):
        # Qwen2VLChat's internal prompt.py:24 only passes one argument ('line') to dump_image_func.
        # So our wrapper must only accept that one argument.
        # We then call the original `dataset_utils.dump_image` with `line_dict_from_model`
        # and the `img_root` which is already defined in the outer scope of `run_qwen_pope_inference`.
        return dump_image(line_dict_from_model, img_root)
    # --- END CRITICAL FIX ---


    # Set up CoT prompt
    cot_prompt = ""
    if args.use_cot:
        cot_prompt = args.cot_prompt if args.cot_prompt else " If you are uncertain or the problem is too complex, make a reasoned guess based on the information provided. Avoid repeating steps indefinitely—provide your best guess even if unsure. Determine whether to think step by step based on the difficulty of the question, considering all relevant information before answering."
        print(f"Using CoT prompt: {cot_prompt}")

    # Initialize Qwen2VLChat model
    print(f"Loading Qwen2.5-VL model from {args.model_path}")
    model = Qwen2VLChat(
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_custom_prompt=True,
        min_pixels=1280*28*28,
        max_pixels=5120*28*28,
        use_image_segmentation=args.use_image_segmentation
    )
    # --- CRITICAL FIX ---
    # Un-comment this line and pass the correctly defined wrapper function.
    model.set_dump_image(qwen_dump_image_wrapper)
    # --- CRITICAL FIX END ---


    # Run inference
    ans_file = open(args.answers_file, "w")
    for i, q_line in tqdm(enumerate(questions), total=len(questions), desc="Running inference"):
        try:
            image_filename = q_line["image"]
            question_text = q_line["text"]
            question_id = q_line["question_id"]

            # Construct input dict for Qwen2VLChat.build_prompt
            qwen_input_dict = {
                'image_path': os.path.join(img_root, image_filename), # Full path to the image
                'question': question_text,
                'task': 'pope',
                'index': question_id
            }

            # Build prompt using Qwen2VLChat's method
            # The dataset argument to build_prompt will be passed down to _build_mmmu_prompt as 'dataset'
            messages = model.build_prompt(qwen_input_dict, dataset="pope")

            # Add CoT prompt if enabled
            if args.use_cot and len(messages) > 0 and messages[-1]['type'] == 'text':
                messages[-1]['value'] += cot_prompt

            # Generate response
            response = model.generate(messages)

            # Prepare result in LLaVA's answers.jsonl format
            ans_id = shortuuid.uuid()
            result_entry = {
                "question_id": question_id,
                "prompt": question_text,
                "text": response,
                "answer_id": ans_id,
                "model_id": args.model_path.split('/')[-1],
                "metadata": {}
            }
            ans_file.write(json.dumps(result_entry) + '\n')
            ans_file.flush()

        except Exception as e:
            warnings.warn(f"Error processing question ID {q_line.get('question_id', 'N/A')}, Image: {q_line.get('image', 'N/A')}. Error: {e}")
            traceback.print_exc()
            continue

    ans_file.close()
    print(f"Inference completed. Results saved to {args.answers_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen2.5-VL POPE Evaluation Inference Script")

    parser.add_argument("--model-path", type=str, required=True, help="Path to the Qwen2.5-VL model")
    parser.add_argument("--image-folder", type=str, required=True, help="Path to the folder containing POPE images.")
    parser.add_argument("--question-file", type=str, required=True, help="Path to the POPE question JSONL file.")
    parser.add_argument("--answers-file", type=str, default="qwen25_vl_pope_answers.jsonl", help="Output file path for generated answers.")

    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--top-p", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)

    parser.add_argument("--use-cot", action="store_true")
    parser.add_argument("--cot-prompt", type=str, default="")

    parser.add_argument("--use-image-segmentation", action="store_true", default=False)
    parser.add_argument("--yolo-model_path", type=str, default="./checkpoints/yolov/yolov8l-seg.pt")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    args = parser.parse_args()

    if args.use_image_segmentation and not os.path.exists(os.path.expanduser(args.yolo_model_path)):
        warnings.warn(f"YOLO model path '{args.yolo_model_path}' does not exist. Disabling image segmentation.")
        args.use_image_segmentation = False

    run_qwen_pope_inference(args)