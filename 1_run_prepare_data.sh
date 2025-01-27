#!/bin/bash

cd phase1/centralized

# Set variables
#DATA_DIR="/data/falcetta/A2V_experiments/OUTPUT_phase1/" # OUTPUT DIRECTORY 
DATA_DIR="/data/falcetta/brain_data/A2V_experiments/OUTPUT_phase1_IXI/" # OUTPUT DIRECTORY
echo "Output directory: ${DATA_DIR}"


SRC_DIR="/data/falcetta/A2V_experiments/OASIS_preprocessed/preprocess_OASIS"
#TGT_DIR="/data/falcetta/A2V_experiments/TOPCOW_preprocessed/preprocess_TopCow_all"
TGT_DIR="/data/falcetta/A2V_experiments/IXI_preprocessed/preprocess_IXI"
SIZE=512 # OUTPUT SIZE

# Run the Python script
python prepare_data.py --out ${DATA_DIR} --size ${SIZE} --src_path ${SRC_DIR}/train/ --tgt_path ${TGT_DIR}/train/


echo "Data preparation done!"