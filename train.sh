#!/usr/bin/env bash

TRAIN_SCRIPT="./train.py"

DATAROOT="./datasets/example"
NAME="latent-example-new"
LORA_A_DIR="./lora/exampleA"
LORA_B_DIR="./lora/exampleB"
CYCLE_GAN_DIR="./checkpoints/example/"

N_EPOCHS=40 
N_EPOCHS_DELAY=10 

LAMBDA_A=10 
LAMBDA_B=10 
LAMBDA_IDENTITY=10 
LAMBDA_GAN=1

ALPHA_GAN=0 
ALPHA_A=0.1
ALPHA_B=0.1

LR=0.0001

python $TRAIN_SCRIPT \
    --dataroot $DATAROOT \
    --name $NAME \
    --model latent_cycle_gan \
    --lora_A_dir $LORA_A_DIR \
    --lora_B_dir $LORA_B_DIR \
    --batch_size 16 \
    --init_with_cycle_gan True \
    --cycle_gan_dir $CYCLE_GAN_DIR \
    --n_epochs $N_EPOCHS \
    --n_epochs_decay $N_EPOCHS_DELAY \
    --lambda_A $LAMBDA_A \
    --lambda_B $LAMBDA_B \
    --lambda_identity $LAMBDA_IDENTITY \
    --lambda_gan $LAMBDA_GAN \
    --alpha_gan $ALPHA_GAN \
    --alpha_A $ALPHA_A \
    --alpha_B $ALPHA_B \
    --lr $LR