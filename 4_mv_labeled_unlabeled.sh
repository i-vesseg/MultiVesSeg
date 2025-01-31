#!/bin/bash



cd phase2/

# Just a reminder to move the labeled slices to the labeled folder
# And the unlabeled slices to the unlabeled folder


# TGT_dir = The one created at the end of PREPROCESSING NOTEBOOK
mkdir ${TGT_dir}/train/labeled
mkdir ${TGT_dir}/train/unlabeled
mv ${TGT_dir}/train/*.npy ${TGT_dir}/train/unlabeled # Move all the slices to the unlabeled folder

# Choose 3 (or more) random annotated slices and move them to the labeled folder
mv ${TGT_dir}/train/unlabeled/${ID_1}_slice* ${TGT_dir}/train/labeled
mv ${TGT_dir}/train/unlabeled/${ID_2}_slice* ${TGT_dir}/train/labeled
mv ${TGT_dir}/train/unlabeled/${ID_3}_slice* ${TGT_dir}/train/labeled
# The more labeled slices and the the more variety in the labeled slices, the better the model will adapt to the new domain