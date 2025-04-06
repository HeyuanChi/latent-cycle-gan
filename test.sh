#!/usr/bin/env bash

TEST_SCRIPT="./test.py"

python $TEST_SCRIPT \
    --dataroot /root/autodl-tmp/pytorch-CycleGAN-and-pix2pix/datasets/cat2dog \
    --name latent-cat2dog \
    --model latent_cycle_gan \
    --num_test 1000