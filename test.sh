#!/usr/bin/env bash

TEST_SCRIPT="./test.py"

python $TEST_SCRIPT \
    --dataroot ./datasets/example \
    --name latent-example \
    --model latent_cycle_gan \
    --num_test 10