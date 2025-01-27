#!/bin/bash

cd phase2/

# Set variables
SRC_EXP_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow" # OUTPUT DIRECTORY
SRC_EXP_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase2/IXI" # OUTPUT DIRECTORY

BATCH_SIZE=8
MAX_STEPS=15000
LABEL_NC=3

#PHASE1_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase1/"
#STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/020000.pt"

PHASE1_DIR="../phase1/centralized"
#STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/generator_HQSWI.pt"
#STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/generator_ToPCoW.pt"
STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/generator_IXI.pt"

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
    --workers=12 \
    #--val_interval= \
    #--save_interval=1000 \
