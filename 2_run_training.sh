#!/bin/bash

#SKIP IF YOU ALREADY HAVE THE PRETRAINED ENCODER
cd phase1/centralized

# Set variables

# See 1_run_prepare_data.sh DATA_DIR
DATA_DIR="/data/falcetta/brain_data/A2V_experiments/OUTPUT_phase1_IXI/" # OUTPUT DIRECTORY
SIZE=512 

# Other variables
N_SAMPLE=8
ITER=250000
AUGMENT="--augment"
BATCH=2

# Run the distributed training script
python -m torch.distributed.launch --nproc_per_node=2 train.py ${DATA_DIR} --size ${SIZE} --n_sample ${N_SAMPLE} --iter ${ITER} ${AUGMENT} --batch ${BATCH} \
        --ckpt_save_dir ${DATA_DIR}/checkpoint \
        --ckpt /home/falcetta/GRENOBLE/MultiVesSeg/phase1/centralized/checkpoint/generator_IXI.pt # Checkpoint to start from
