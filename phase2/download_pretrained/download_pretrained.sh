#!/bin/bash

pip install gdown

# Create the target directory if it doesn't exist
OUTPUT_FILE="phase2/pretrained_models"

mkdir -p ${OUTPUT_FILE}

# URLs of the pretrained models
MODEL_URLS=(
    "https://github.com/richzhang/PerceptualSimilarity/raw/refs/heads/master/lpips/weights/v0.1/alex.pth"
    "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
)

# Filenames to check for and download as
MODEL_NAMES=(
    "alex.pth"
    "alex_pretr.pth"  # Renamed file
)

# Download each model if not already downloaded
for i in "${!MODEL_URLS[@]}"; do
    FILE_PATH="${OUTPUT_FILE}/${MODEL_NAMES[i]}"
    
    if [ -f "$FILE_PATH" ]; then
        echo "${MODEL_NAMES[i]} already exists, skipping download."
    else
        wget -P ${OUTPUT_FILE} "${MODEL_URLS[i]}"
        if [ "${MODEL_NAMES[i]}" == "alex_pretr.pth" ]; then
            mv "${OUTPUT_FILE}/alexnet-owt-7be5be79.pth" "$FILE_PATH"
        fi
    fi
done

# Google Drive file information
FILE_ID="1coFTz-Kkgvoc_gRT8JFzqCgeC3lAFWQp"
BACKBONE_FILE="${OUTPUT_FILE}/backbone.pth"

# Download the file from Google Drive if it doesn't exist
if [ -f "$BACKBONE_FILE" ]; then
    echo "backbone.pth already exists, skipping download."
else
    gdown https://drive.google.com/uc?id=${FILE_ID} -O ${BACKBONE_FILE}
fi

echo "Download completed. Models are saved in ${OUTPUT_FILE}"
