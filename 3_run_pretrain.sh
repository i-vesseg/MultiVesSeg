#!/bin/bash

cd phase2/

# Set variables
SRC_EXP_DIR="OUTPUT_TEST/TopCow" # OUTPUT DIRECTORY

BATCH_SIZE=8
MAX_STEPS=15000
LABEL_NC=3

PHASE1_DIR="../phase1/centralized"
STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/final_checkpoint.pt"
STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/020000.pt"
STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/generator_IXI.pt"
STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/generator_HQSWI.pt"
STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/generator_ToPCoW.pt"

SRC_LABEL=0

# Run the Python script
python scripts/pretrain.py \
    --exp_dir=${SRC_EXP_DIR} \
    --batch_size=${BATCH_SIZE} \
    --start_from_latent_avg \
    --label_nc=${LABEL_NC} \
    --max_steps=${MAX_STEPS} \
    --stylegan_weights=${STYLEGAN_WEIGHTS} \
    --only_intra \
    --src_label=${SRC_LABEL} \
    --n_domains=2 \
    --save_interval=100 \
    --val_interval=1 \
    --workers=12 \
