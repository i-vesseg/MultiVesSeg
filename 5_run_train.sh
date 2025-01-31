#!/bin/bash

cd phase2/

# Set variables
#SRC_EXP_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow" # OUTPUT DIRECTORY
#TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow_TARGET" # OUTPUT DIRECTORY

SRC_EXP_DIR="/home/geninana/data_ssd/DqnieleF/A2V_experiments/ADAPTATION/TOF" # OUTPUT DIRECTORY FROM 3_run_pretrain.sh
TGT_exp_dir="/home/geninana/data_ssd/DqnieleF/A2V_experiments/ADAPTATION/TOF_ADAPT" # OUTPUT DIRECTORY for this script

BATCH_SIZE=32 # Default:8 but you can increase this value if you have more GPU memory
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