#!/bin/bash

cd phase2/
# Set variables
SRC_EXP_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase2/IXI/checkpoints"
SRC_EXP_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow/checkpoints" # OUTPUT DIRECTORY

SRC_FAKE_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase2/TopCow_FAKE/fake_source" # OUTPUT DIRECTORY

SRC_LABEL=0

# Run the Python script
python generate_volumes.py \
        --ckpt_dir=${SRC_EXP_DIR} \
        --label=${SRC_LABEL} \
        --n_volumes=20 \
        --slices_in_one_z=10 \
        --n_segmentation_labels=3 \
        --out_dir=${SRC_FAKE_DIR} \
        --n_domains=2