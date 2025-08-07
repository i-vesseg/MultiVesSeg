#!/usr/bin/env python3

import argparse
import os
import nibabel as nib
import numpy as np
from nipype.interfaces import fsl
from nipype.interfaces.ants import BrainExtraction
import logging
import sys
import subprocess
from HD_BET.run import run_hd_bet
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def generate_output_path(input_path, method, output_dir=None):
    """
    Generate output path based on input path
    """
    input_path = Path(input_path)
    if output_dir:
        output_dir = Path(output_dir)
        # Preserve directory structure relative to input directory
        if input_path.is_file():
            relative_path = input_path.parent.name
        else:
            relative_path = input_path.relative_to(input_path.parent)
        directory = output_dir / relative_path
    else:
        directory = input_path.parent

    filename = input_path.name
    base = input_path.stem
    if base.endswith('.nii'):  # Handle .nii.gz
        base = base[:-4]
    
    output_filename = f"{base}_skull_stripped_{method}.nii.gz"
    directory.mkdir(parents=True, exist_ok=True)
    return str(directory / output_filename)

def find_nifti_files(directory):
    """
    Recursively find all NIFTI files in directory
    """
    nifti_files = []
    for ext in ['.nii', '.nii.gz']:
        nifti_files.extend(Path(directory).rglob(f'*{ext}'))
    return nifti_files

def skull_strip_hdbet(input_path, output_path, device='gpu'):
    """
    Perform skull stripping using HD-BET
    """
    try:
        mode = 'fast'  # or 'accurate' for better results but slower
        device = 'cpu' if device.lower() == 'cpu' else 'gpu'
        
        run_hd_bet(
            input_path,
            output_path,
            mode=mode,
            device=device,
            tta=0,
            pp=True
        )
        return output_path
        
    except Exception as e:
        raise RuntimeError(f"HD-BET skull stripping failed: {str(e)}")

def skull_strip_fsl(input_path, output_path, modality='t1'):
    """
    Perform skull stripping using FSL's BET
    """
    bet = fsl.BET()
    bet.inputs.in_file = input_path
    bet.inputs.out_file = output_path
    bet.inputs.frac = 0.5
    
    if modality.lower() == 'ct':
        bet.inputs.frac = 0.3
        bet.inputs.robust = True
    
    bet.run()
    return output_path

def skull_strip_ants(input_path, output_path, modality='t1'):
    """
    Perform skull stripping using ANTs
    """
    bex = BrainExtraction()
    bex.inputs.dimension = 3
    bex.inputs.anatomical_image = input_path
    bex.inputs.brain_template = 'template.nii.gz'
    bex.inputs.brain_probability_mask = 'probability_mask.nii.gz'
    bex.inputs.out_prefix = 'ants_'
    
    if modality.lower() == 'ct':
        bex.inputs.num_threads = 4
        bex.inputs.keep_temporary_files = 0
    
    bex.run()
    return output_path

def process_single_file(args, input_path, logger):
    """
    Process a single NIFTI file
    """
    try:
        output_path = generate_output_path(input_path, args.method, args.output_dir)
        
        logger.info(f"Processing: {input_path}")
        logger.info(f"Output will be saved to: {output_path}")
        
        if args.method == 'fsl':
            output_path = skull_strip_fsl(str(input_path), output_path, args.modality)
        elif args.method == 'ants':
            output_path = skull_strip_ants(str(input_path), output_path, args.modality)
        else:  # hdbet
            output_path = skull_strip_hdbet(str(input_path), output_path, args.device)
        
        logger.info(f"Completed processing: {input_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        return False

def validate_input(path):
    """
    Validate input path exists
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input path {path} does not exist")

def main():
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description='Skull stripping for brain MRA/CT images')
    parser.add_argument('input_path', help='Input NIFTI file or directory path')
    parser.add_argument('--output_dir', help='Output directory (optional)')
    parser.add_argument('--modality', choices=['mra', 'ct'], default='mra',
                      help='Image modality (default: mra)')
    parser.add_argument('--method', choices=['fsl', 'ants', 'hdbet'], default='hdbet',
                      help='Skull stripping method (default: hdbet)')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu',
                      help='Device to use for HD-BET (default: gpu)')
    parser.add_argument('--max_workers', type=int, default=1,
                      help='Maximum number of parallel processes (default: 1)')
    
    args = parser.parse_args()
    
    try:
        validate_input(args.input_path)
        start_time = time.time()
        
        if os.path.isfile(args.input_path):
            # Single file processing
            if not args.input_path.endswith(('.nii', '.nii.gz')):
                raise ValueError("Input file must be in NIFTI format (.nii or .nii.gz)")
            success = process_single_file(args, Path(args.input_path), logger)
            total_files = 1
            successful_files = 1 if success else 0
            
        else:
            # Directory processing
            input_files = find_nifti_files(args.input_path)
            total_files = len(input_files)
            
            if total_files == 0:
                logger.warning(f"No NIFTI files found in {args.input_path}")
                return
            
            logger.info(f"Found {total_files} NIFTI files to process")
            
            # Process files in parallel
            successful_files = 0
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(process_single_file, args, f, logger) 
                          for f in input_files]
                successful_files = sum(future.result() for future in futures)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Successfully processed {successful_files} out of {total_files} files")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    
    
    
# # Process a single file
# docker run your-image skull_strip.py input.nii.gz

# # Process a directory
# docker run your-image skull_strip.py /input/directory

# # Process a directory with custom output location
# docker run your-image skull_strip.py /input/directory --output_dir /output/directory

# # Process a directory with parallel processing
# docker run your-image skull_strip.py /input/directory --max_workers 4

# # Full example with all options
# docker run your-image skull_strip.py /input/directory \
#     --output_dir /output/directory \
#     --method hdbet \
#     --device gpu \
#     --modality ct \
#     --max_workers 4


# docker run -v /path/to/input:/input \
#            -v /path/to/output:/output \
#            your-image skull_strip.py /input --output_dir /output