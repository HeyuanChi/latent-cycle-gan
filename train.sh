#!/usr/bin/env bash

TRAIN_SCRIPT="./train.py"

python $TRAIN_SCRIPT \
    --dataroot /root/autodl-tmp/pytorch-CycleGAN-and-pix2pix/datasets/cat2dog \
    --name latent-cat2dog \
    --model latent_cycle_gan \
    --lora_A_dir /root/autodl-tmp/diffusers/outputs/output-cat/checkpoint-5000 \
    --lora_B_dir /root/autodl-tmp/diffusers/outputs/output-dog/checkpoint-5000 \
    --batch_size 16 \
    --init_with_cycle_gan True \
    --cycle_gan_dir /root/autodl-tmp/pytorch-CycleGAN-and-pix2pix/checkpoints/cat2dog/ \
    --n_epochs 40 \
    --n_epochs_decay 10 \
    --alpha_gan 1 \
    --alpha_A 0.1 \
    --alpha_B 0.1 \
    --lr 0.0001