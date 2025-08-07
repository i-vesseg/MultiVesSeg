# Docker Tutorial for Brain Skull Stripping Tool

## 1. Project Structure
First, create a directory for your project with the following structure:
```
skull-strip/
├── Dockerfile
├── requirements.txt
├── entrypoint.py
└── skull_strip.py
```

## 2. File Contents
Create or copy the following files:

### requirements.txt
```
nibabel
nipype
numpy
HD-BET
torch
```

### entrypoint.py and skull_strip.py
Copy the previously provided code into these files.

## 3. Building the Docker Image
Navigate to your project directory and build the image:
```bash
cd skull-strip
docker build -t skull-stripper:latest .
```

## 4. Running the Container

### A. Single File Processing
```bash
# Basic usage with a single file
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
    skull-stripper:latest skull_strip.py /input/brain.nii.gz --output_dir /output

# Using HD-BET with GPU
docker run --gpus all -v $(pwd)/input:/input -v $(pwd)/output:/output \
    skull-stripper:latest skull_strip.py /input/brain.nii.gz \
    --method hdbet --device gpu --output_dir /output
```

### B. Directory Processing
```bash
# Process all NIFTI files in a directory
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
    skull-stripper:latest skull_strip.py /input --output_dir /output

# Process with parallel execution
docker run --gpus all -v $(pwd)/input:/input -v $(pwd)/output:/output \
    skull-stripper:latest skull_strip.py /input \
    --output_dir /output --max_workers 4
```

## 5. Common Use Cases

### CT Images
```bash
docker run --gpus all -v $(pwd)/input:/input -v $(pwd)/output:/output \
    skull-stripper:latest skull_strip.py /input \
    --modality ct --method hdbet --output_dir /output
```

### MRA Images
```bash
docker run --gpus all -v $(pwd)/input:/input -v $(pwd)/output:/output \
    skull-stripper:latest skull_strip.py /input \
    --modality mra --method hdbet --output_dir /output
```

### Using FSL Instead of HD-BET
```bash
docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
    skull-stripper:latest skull_strip.py /input \
    --method fsl --output_dir /output
```

## 6. Tips and Troubleshooting

### Directory Permissions
If you encounter permission issues:
```bash
# Create output directory with proper permissions
mkdir -p output
chmod 777 output

# Run docker with explicit user mapping
docker run --user $(id -u):$(id -g) ...
```

### GPU Access
Make sure you have NVIDIA Container Toolkit installed for GPU support:
```bash
# Check GPU access
docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
```

### Memory Issues
If you encounter memory issues with large datasets:
```bash
# Limit memory usage
docker run --memory=8g --memory-swap=8g ...
```

## 7. Parameters Reference

Available parameters for skull_strip.py:
- `--method`: Choose between 'hdbet' (default), 'fsl', or 'ants'
- `--modality`: Choose between 'mra' (default) or 'ct'
- `--device`: Choose between 'gpu' (default) or 'cpu'
- `--max_workers`: Number of parallel processes (default: 1)
- `--output_dir`: Output directory path

## 8. Example Directory Structure
```
your_project/
├── input/
│   ├── patient1/
│   │   ├── scan1.nii.gz
│   │   └── scan2.nii.gz
│   └── patient2/
│       └── scan1.nii.gz
└── output/
    ├── patient1/
    │   ├── scan1_skull_stripped_hdbet.nii.gz
    │   └── scan2_skull_stripped_hdbet.nii.gz
    └── patient2/
        └── scan1_skull_stripped_hdbet.nii.gz
```
