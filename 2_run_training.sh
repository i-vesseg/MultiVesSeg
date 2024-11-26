#!/bin/bash

cd phase1/centralized

# Set variables

# See 1_run_prepare_data.sh
DATA_DIR="../OUTPUT/" 
SIZE=512 

# Other variables
N_SAMPLE=8
ITER=250000
AUGMENT="--augment"

# Run the distributed training script
python -m torch.distributed.launch --nproc_per_node=2 train.py ${DATA_DIR} --size ${SIZE} --n_sample ${N_SAMPLE} --iter ${ITER} ${AUGMENT} --batch 2
