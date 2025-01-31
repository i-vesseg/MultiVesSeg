#!/bin/bash

cd phase2/

# Set variables
PHASE1_DIR="../phase1/centralized"


############################### CT ########################################
#SRC_EXP_DIR="/home/geninana/data_ssd/DqnieleF/A2V_experiments/ADAPTATION/CT" # OUTPUT DIRECTORY
#STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/generator_ToPCoW.pt"

############################### TOF ########################################
SRC_EXP_DIR="/home/geninana/data_ssd/DqnieleF/A2V_experiments/ADAPTATION/TOF" # OUTPUT DIRECTORY
STYLEGAN_WEIGHTS="${PHASE1_DIR}/checkpoint/generator_IXI.pt"
###########################################################################


BATCH_SIZE=32 # Default:8 but you can increase this value if you have more GPU memory
MAX_STEPS=15000 # Number of training steps
LABEL_NC=3

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
