#!/bin/bash

deepspeed ./self_filter_qwen/filtering/train.py \
    --feature_extractor_setting clip \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoint/Qwen2-VL-7B-Instruct  \
    --data_path ./data/llava_instruct_80k_add_idx.json \
    --image_folder ./data/cocodataset/train2017 \
    --bf16 True \
    --output_dir ./checkpoint/qwen_result \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.02 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
