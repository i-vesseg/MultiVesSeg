#!/bin/bash

cd phase2/
# ----------------------------
# ----------------------------
# TGT_exp_dir="/home/geninana/data_ssd/DqnieleF/MODELW/TopCow_TARGET" # SAVED MODEL DIRECTORY
# INFO_path="/home/geninana/data_ssd/DqnieleF/MultiVesSeg/preprocessing/info_CT_TAS_GRENOBLE_TEST.pkl" # (From the preprocessing notebook)
# TEST_dir="/home/geninana/data_ssd/DqnieleF/A2V_experiments/CT_TAS_GRENOBLE_TEST_preprocessed/OUTPUTS" # OUTPUT DIRECTORY
# # ----------------------------
# TGT_exp_dir="/home/geninana/data_ssd/DqnieleF/MODELW/IXI_TARGET" # SAVED MODEL DIRECTORY
# INFO_path="/home/geninana/data_ssd/DqnieleF/MultiVesSeg/preprocessing/info_TOF_GRENOBLE.pkl" # (From the preprocessing notebook)
# TEST_dir="/home/geninana/data_ssd/DqnieleF/A2V_experiments/TOF_GRENOBLE_preprocessed/OUTPUTS" # OUTPUT DIRECTORY

# ---------------------------- UPDATE THESE PATHS
TGT_exp_dir="/home/geninana/data_ssd/DqnieleF/MODELW/checkpoint_TopCow_tgt_9slices" # SAVED MODEL DIRECTORY
INFO_path="/home/geninana/data_ssd/DqnieleF/MultiVesSeg/preprocessing/info_CT_GRENOBLE_5.pkl" # (INFO FILE From the preprocessing notebook)
TEST_dir="/home/geninana/data_ssd/DqnieleF/A2V_experiments/CT_TEST_5/OUTPUTS9" # OUTPUT DIRECTORY



python scripts/inference.py \
        --metadata=${INFO_path} \
        --exp_dir=${TEST_dir} \
        --start_from_latent_avg \
        --label_nc=3 \
        --checkpoint_dir=${TGT_exp_dir}/checkpoints \
        --src_label 0 \
        --tgt_label 1 \
        --n_domains=2 
