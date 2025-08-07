#!/bin/bash

cd phase2/

# TRAINED ON TOPCOW or IXI
TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/IXI_TARGET" # OUTPUT DIRECTORY
TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow_TARGET" # (From the previous script + TIMESTAMP)

# ----------------------------

INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_TopCow_all.pkl" # (From the preprocessing notebook)
TEST_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow_INFERENCE" # OUTPUT DIRECTORY


INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_TOPCOW_24MR_TEST.pkl"
TEST_dir='/data/falcetta/TOPCOW_RES/A2V/24/MR/RESULTS'

INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_TOPCOW_23MR_TEST.pkl"
TEST_dir='/data/falcetta/TOPCOW_RES/A2V/23/MR/RESULTS'

INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_TOPCOW_23CT_TEST.pkl"
TEST_dir='/data/falcetta/TOPCOW_RES/A2V/23/CT/RESULTS'

INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_TOPCOW_24CT_TEST.pkl"
TEST_dir='/data/falcetta/TOPCOW_RES/A2V/24/CT/RESULTS'

# ----------------------------


python scripts/inference.py \
        --metadata=${INFO_path} \
        --exp_dir=${TEST_dir} \
        --start_from_latent_avg \
        --label_nc=3 \
        --checkpoint_dir=${TGT_exp_dir}/checkpoints \
        --src_label 0 \
        --tgt_label 1 \
        --n_domains=2 
