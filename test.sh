#!/usr/bin/env bash

TEST_SCRIPT="./test.py"

python $TEST_SCRIPT \
    --dataroot ./datasets/example \
    --name latent-example \
    --model latent_example \
    --num_test 10