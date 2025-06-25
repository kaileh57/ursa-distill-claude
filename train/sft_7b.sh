#!/bin/bash
# Training script for 7B model - optimized for single node

uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-7B-Instruct"
lr=2e-5  # Slightly higher LR for 7B
epochs=5
weight_decay=1e-4
micro_batch_size=2  # Larger batch size for 7B
gradient_accumulation_steps=2
gpu_count=$(nvidia-smi -L | wc -l)

echo "Training Ursa Minor 7B with ${gpu_count} GPUs"

torchrun --nproc-per-node ${gpu_count} --master_port 12345 \
    train/sft_7b.py \
    --block_size=16384 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="ursa-minor/s1K-7b-tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="configs/training/fsdp_config_qwen_7b.json" \
    --bf16=True \
    --eval_strategy="no" \
    --logging_steps=1 \
    --save_strategy="epoch" \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="checkpoints/ursa-minor-7b-${uid}" \
    --push_to_hub=true \
    --hub_model_id="ursa-minor/7b-${uid}" \
    --save_only_model=True \
    --gradient_checkpointing=True \
    --report_to="wandb" \
    --run_name="ursa-minor-7b-${uid}"