#!/usr/bin/env bash

TRAIN_SCRIPT="./train.py"

python $TRAIN_SCRIPT \
    --dataroot ./datasets/example \
    --name latent-example-new \
    --model latent_cycle_gan \
    --lora_A_dir ./lora/exampleA \
    --lora_B_dir ./lora/exampleB \
    --batch_size 16 \
    --init_with_cycle_gan True \
    --cycle_gan_dir ./checkpoints/example/ \
    --n_epochs 40 \
    --n_epochs_decay 10 \
    --alpha_gan 1 \
    --alpha_A 0.1 \
    --alpha_B 0.1 \
    --lr 0.0001