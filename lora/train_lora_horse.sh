#!/usr/bin/env bash

RESOLUTION=512
BATCH_SIZE=8
GRAD_ACCUM=4
LR=1e-5
MAX_STEPS=5000

LORA_RANK=8

TRAIN_SCRIPT="./train_dreambooth_lora.py"


python $TRAIN_SCRIPT \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --instance_data_dir="/root/autodl-tmp/pytorch-CycleGAN-and-pix2pix/datasets/horse2zebra/trainA" \
  --output_dir="/root/autodl-tmp/diffusers/outputs/output-horse" \
  --instance_prompt="a photo of horse" \
  --resolution=$RESOLUTION \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GRAD_ACCUM \
  --learning_rate=$LR \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$MAX_STEPS \
  --rank=$LORA_RANK