#!/bin/bash

cd phase2/

# Set variables
#SRC_EXP_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow" # OUTPUT DIRECTORY
#TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow_TARGET" # OUTPUT DIRECTORY

SRC_EXP_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase2/IXI" # OUTPUT DIRECTORY
TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/IXI_TARGET" # OUTPUT DIRECTORY

BATCH_SIZE=8
MAX_STEPS=15000


# Run the Python script
python scripts/train.py \
    --exp_dir=${TGT_exp_dir} \
    --start_from_latent_avg \
    --label_nc=3 \
    --max_steps=20000 \
    --checkpoint_dir=${SRC_EXP_DIR}/checkpoints  \
    --one_target_slice \
    --src_label 0 \
    --tgt_label 1 \
    --n_domains=2 \
    --workers=12 \
    #--save_interval=1 \