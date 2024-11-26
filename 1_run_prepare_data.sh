#!/bin/bash

cd phase1/centralized

# Set variables
DATA_DIR="../OUTPUT/" # OUTPUT DIRECTORY
echo "Output directory: ${DATA_DIR}"


SRC_DIR="/data/falcetta/brain_data/A2V_PREPROCESSED/preprocess_OASIS"
TGT_DIR="/home/falcetta/GRENOBLE/MultiVesSeg/preprocessing/preprocess_TopCow_FULL_SEG_1"
SIZE=512 # OUTPUT SIZE

# Run the Python script
python prepare_data.py --out ${DATA_DIR} --size ${SIZE} --src_path ${SRC_DIR}/train/ --tgt_path ${TGT_DIR}/train/


echo "Data preparation done!"