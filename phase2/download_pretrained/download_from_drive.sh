#!/bin/bash

#Install gdown if not already installed
pip install gdown

# Create the target directory if it doesn't exist
OUTPUT_FILE="output/folder"

mkdir -p ${OUTPUT_FILE}

# Google Drive file information
FILE_ID="1coFTz-Kkgvoc_gRT8JFzqCgeC3lAFWQp"
FILEPATH="${OUTPUT_FILE}/filename.aaa"

# Download the file from Google Drive if it doesn't exist
if [ -f "$FILEPATH" ]; then
    echo $FILEPATH already exists, skipping download.
else
    gdown https://drive.google.com/uc?id=${FILE_ID} -O ${FILEPATH}
fi

echo "Download completed. Models are saved in ${OUTPUT_FILE}"
