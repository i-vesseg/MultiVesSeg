#!/bin/bash

cd phase2/

# Set variables
TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow_TARGET" # (From the previous script + TIMESTAMP)
INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_TopCow_all.pkl" # (From the preprocessing notebook)

TEST_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow_INFERENCE" # OUTPUT DIRECTORY

# ----------------------------

TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/IXI_TARGET"
INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_IXI.pkl" # (From the preprocessing notebook)

TEST_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/IXI_INFERENCE" # OUTPUT DIRECTORY
# ----------------------------

TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/IXI_TARGET"
INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_ITKTubeTK_A2V.pkl"
TEST_dir="/data/galati/brain_data/ITKTubeTK/fold1/imagesTs_A2V_PRED"

# ----------------------------
TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/IXI_TARGET"
INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_LONDON_TEST.pkl"
TEST_dir="/data/falcetta/LONDON_TEST/OUT_A2V"

# ----------------------------
TGT_exp_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/GRENOBLE_CTs_TARGET_UNLABELED_VAL"
INFO_path="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/info_GrenobleCTs.pkl"
TEST_dir="/data/falcetta/A2V_experiments/OUTPUT_phase2/GRENOBLE_CTs_TARGET_UNLABELED_VAL_INFERENCE"


python scripts/inference.py \
        --metadata=${INFO_path} \
        --exp_dir=${TEST_dir} \
        --start_from_latent_avg \
        --label_nc=3 \
        --checkpoint_dir=${TGT_exp_dir}/checkpoints \
        --src_label 0 \
        --tgt_label 1 \
        --n_domains=2 \
        #--compute_final_metrics=False 
