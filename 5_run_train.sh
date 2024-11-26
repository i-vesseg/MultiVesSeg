#!/bin/bash

cd phase2/

# Set variables
SRC_exp_dir="OUTPUT_TEST/TopCow" # OUTPUT DIRECTORY from 3_run_pretrain.sh
TGT_exp_dir="OUTPUT_TEST/TopCow_TRAIN" # OUTPUT DIRECTORY

BATCH_SIZE=8
MAX_STEPS=15000
LABEL_NC=3


# Run the Python script
python scripts/train.py \
    --exp_dir=${TGT_exp_dir} \
    --start_from_latent_avg \
    --label_nc=3 \
    --max_steps=20000 \
    --checkpoint_dir=${SRC_exp_dir}/checkpoints  \
    --one_target_slice \
    --src_label 0 \
    --tgt_label 1 \
    --n_domains=2 \
    --save_interval=1 \
    --workers=12 \