#!/bin/bash
AUTO_DL_TMP="$HOME/autodl-tmp"

python -m llava.eval.model_vqa_loader \
    --model-path ~/.cache/huggingface/hub/models--imagecache--llava-v1.5-7b-lora-noprefusion \
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder $AUTO_DL_TMP/playground/data/eval/pope/val2014 \
    --answers-file $AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-lora-noprefusion.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir $AUTO_DL_TMP/playground/data/eval/pope/coco \
    --question-file $AUTO_DL_TMP/playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file $AUTO_DL_TMP/playground/data/eval/pope/answers/llava-v1.5-7b-lora-noprefusion.jsonl

