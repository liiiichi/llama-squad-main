#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 python train_llama_squad.py \
    --model_name togethercomputer/Llama-2-7B-32K-Instruct \
    --dataset_name data/squad_v2 \
    --bf16 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_steps 10000 \
    --merge_and_push \
    --save_steps 1000 \
    --learning_rate=2e-7
