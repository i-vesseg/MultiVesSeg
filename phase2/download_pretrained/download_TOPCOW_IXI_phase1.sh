#!/bin/bash

#Install gdown if not already installed
pip install gdown

# Create the target directory if it doesn't exist
OUTPUT_FILE="phase1/centralized/checkpoint"

mkdir -p ${OUTPUT_FILE}

# Google Drive file information

FILE_ID="1I-0h-FX-FgKQOpaEPVEvwKXB3Y0BOTYI"
FILEPATH="${OUTPUT_FILE}/generator_HQSWI.pt"

FILE_ID="1v_VtLTKO57uYXgSAgHzlZBI_aQUz1g3y"
FILEPATH="${OUTPUT_FILE}/generator_ToPCoW.pt"

FILE_ID="1ce4q_IqkFloKrbMNYTDFo6ENRDgyiha4"
FILEPATH="${OUTPUT_FILE}/generator_IXI.pt"

# Download the file from Google Drive if it doesn't exist
if [ -f "$FILEPATH" ]; then
    echo $FILEPATH already exists, skipping download.
else
    gdown https://drive.google.com/uc?id=${FILE_ID} -O ${FILEPATH}
fi

echo "Download completed. Models are saved in ${OUTPUT_FILE}"
