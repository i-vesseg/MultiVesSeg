#!/bin/bash



cd phase2/

TGT_dir="OUTPUT_TEST/TopCow" # OUTPUT DIRECTORY

mkdir ${TGT_dir}/train/labeled
mkdir ${TGT_dir}/train/unlabeled
mv ${TGT_dir}/train/*.npy ${TGT_dir}/train/unlabeled
mv ${TGT_dir}/train/unlabeled/${ID_1}_slice* ${TGT_dir}/train/labeled
mv ${TGT_dir}/train/unlabeled/${ID_2}_slice* ${TGT_dir}/train/labeled
mv ${TGT_dir}/train/unlabeled/${ID_3}_slice* ${TGT_dir}/train/labeled