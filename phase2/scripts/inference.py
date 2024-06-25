"""
This file runs the main training/val loop
"""
import os
import json
import sys
import pprint
import pickle
import numpy as np
import torch
import nibabel as nib
import glob
import gc

from batchgenerators.augmentations.utils import resize_segmentation
from skimage.transform import resize

sys.path.append(".")
sys.path.append("..")

def check_is_train():
    if os.path.isdir("configs_train") and os.path.isdir("configs_pretrain"):
        os.rename("configs_train", "configs")
        return
    elif os.path.isdir("configs_train") and os.path.isdir("configs"):
        os.rename("configs", "configs_pretrain")
        os.rename("configs_train", "configs")
        return
    elif os.path.isdir("configs_pretrain") and os.path.isdir("configs"):
        return
    else:
        raise Exception("Something is wrong with your configs folder")
check_is_train()

from options import Options
from training.coach_inference import Coach
from configs import data_configs
from utils import data_utils

def get_best_models(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "timestamp.txt"), "r") as file:
        timestamp = file.readlines()

    best_models = [line for line in timestamp if line.startswith("**Best saved at best_model")][-1]
    best_models = eval(best_models.split(", Loss - ")[1])

    return [os.path.join(checkpoint_dir, model_name[0]) for model_name in best_models]

def main():
    opts = Options(is_train=False).parse()
    if os.path.exists(opts.exp_dir):
        raise Exception('Oops... {} already exists'.format(opts.exp_dir))
    os.makedirs(opts.exp_dir)

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)
    
    with open(opts.metadata, "rb") as file:
        info_val = pickle.load(file)["test"]
    def get_slice_size(slc):
        return len(range(*slc.indices(slc.stop)))
    def cropToShape(crop):
        return [get_slice_size(c) for c in crop]
    info_val["shapesAfterCropping"] = [cropToShape(crop) for crop in info_val["crops"]]
    
    ckpts = get_best_models(opts.checkpoint_dir)
    
    all_img_names = set()
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    if opts.only_intra:
        all_img_names.update([p.split("_slice")[0] for p in data_utils.make_dataset(dataset_args['test_source_root'])])
    else:
        all_img_names.update([p.split("_slice")[0] for p in data_utils.make_dataset(dataset_args['test_target_root']["labeled"])])
    
    for curr_img_name in sorted(all_img_names):
        if opts.only_intra:
            data_configs.DATASETS[opts.dataset_type]['test_source_root'] = curr_img_name
        else:
            data_configs.DATASETS[opts.dataset_type]['test_target_root']["labeled"] = curr_img_name
        curr_img_name = os.path.basename(curr_img_name)
        
        intra_pred = 0
        if not opts.only_intra:
            inter_pred = 0
            volume_trans = 0        
        
        for ckpt in ckpts[-1*opts.ensemble_size:]:
            opts.checkpoint_path = ckpt
            global_step = torch.load(ckpt, map_location='cpu')["global_step"]
            coach = Coach(opts, global_step)
        
            for confirm_curr_img_name, pred_dict in coach.infer():
                assert curr_img_name == confirm_curr_img_name
                intra_pred += pred_dict["intra"]
                if not opts.only_intra:
                    inter_pred += pred_dict["intra"]
                    volume_trans = pred_dict["trans"]
            
            del coach
            gc.collect()
            torch.cuda.empty_cache()
    
        intra_pred /= len(ckpts)
        if not opts.only_intra:
            inter_pred /= len(ckpts)
        
        volume_intra = np.argmax(intra_pred, axis=1)
        if not opts.only_intra:
            volume_inter = np.argmax(inter_pred, axis=1)
            volume_ensemble = np.average([inter_pred, intra_pred], axis=0, weights=[0.6, 0.4])
            volume_ensemble = np.argmax(volume_ensemble, axis=1)
        
        idx_patient = [curr_img_name in p for p in info_val["preprocessed_paths"]].index(True)

        #postprocessing
        (z_size, x_size, y_size) = info_val["shapesBeforePadding"][idx_patient]
        volume_intra = volume_intra[
            :, (512-x_size) // 2: (512-x_size) // 2 + x_size, (512-y_size) // 2: (512-y_size) // 2 + y_size
        ]
        if not opts.only_intra:
            volume_inter = volume_inter[
                :, (512-x_size) // 2: (512-x_size) // 2 + x_size, (512-y_size) // 2: (512-y_size) // 2 + y_size
            ]
            volume_ensemble = volume_ensemble[
                :, (512-x_size) // 2: (512-x_size) // 2 + x_size, (512-y_size) // 2: (512-y_size) // 2 + y_size
            ]
            volume_trans = volume_trans[
                :, (512-x_size) // 2: (512-x_size) // 2 + x_size, (512-y_size) // 2: (512-y_size) // 2 + y_size
            ]

        depth = info_val["depths"][idx_patient]
        assert depth == len(volume_intra)
        #if depth != len(volume):
            #volume = resize_segmentation(volume, [depth, *volume.shape[1:]], order=2)        

        first_slice, last_slice = info_val["z_splits"][idx_patient]
        if last_slice == depth:
            last_slice = 0
        volume_intra = np.concatenate([
            np.zeros([first_slice, *volume_intra.shape[1:]], dtype=volume_intra.dtype),
            volume_intra,
            np.zeros([-1*last_slice, *volume_intra.shape[1:]], dtype=volume_intra.dtype),
        ])
        if not opts.only_intra:
            volume_inter = np.concatenate([
                np.zeros([first_slice, *volume_inter.shape[1:]], dtype=volume_inter.dtype),
                volume_inter,
                np.zeros([-1*last_slice, *volume_inter.shape[1:]], dtype=volume_inter.dtype),
            ])
            volume_ensemble = np.concatenate([
                np.zeros([first_slice, *volume_ensemble.shape[1:]], dtype=volume_ensemble.dtype),
                volume_ensemble,
                np.zeros([-1*last_slice, *volume_ensemble.shape[1:]], dtype=volume_ensemble.dtype),
            ])
            volume_trans = np.concatenate([
                np.full([first_slice, *volume_trans.shape[1:]], -1, dtype=volume_trans.dtype),
                volume_trans,
                np.full([-1*last_slice, *volume_trans.shape[1:]], -1, dtype=volume_trans.dtype),
            ])

        #inference (slow down otherwise)
        (x_size, y_size, z_size) = info_val["shapesAfterCropping"][idx_patient]
        volume_intra = resize_segmentation(volume_intra, [z_size, x_size, y_size], order=1)
        if not opts.only_intra:
            volume_inter = resize_segmentation(volume_inter, [z_size, x_size, y_size], order=1)
            volume_ensemble = resize_segmentation(volume_ensemble, [z_size, x_size, y_size], order=1)
            volume_trans = resize(volume_trans, [z_size, x_size, y_size], 3, cval=0, mode='edge', anti_aliasing=False)

        (x_size, y_size, z_size) = info_val["shapes"][idx_patient]
        crop = info_val["crops"][idx_patient]
        final_volume = np.zeros((x_size, y_size, z_size), dtype=volume_intra.dtype)
        final_volume[crop] = volume_intra.transpose(1, 2, 0)
        volume_intra = final_volume
        if not opts.only_intra:
            final_volume = np.zeros((x_size, y_size, z_size), dtype=volume_inter.dtype)
            final_volume[crop] = volume_inter.transpose(1, 2, 0)
            volume_inter = final_volume
            final_volume = np.zeros((x_size, y_size, z_size), dtype=volume_ensemble.dtype)
            final_volume[crop] = volume_ensemble.transpose(1, 2, 0)
            volume_ensemble = final_volume
            final_volume = np.full((x_size, y_size, z_size), -1, dtype=volume_trans.dtype)
            final_volume[crop] = volume_trans.transpose(1, 2, 0)
            volume_trans = final_volume
        
        if opts.do_flip:
            volume_intra = volume_intra[:,::-1]
            if not opts.only_intra:
                volume_inter = volume_inter[:,::-1]
                volume_ensemble = volume_ensemble[:,::-1]
                volume_trans = volume_trans[:,::-1]

        metadata = info_val["metadata"][idx_patient]["brain"]
        if metadata is None:
            metadata = info_val["metadata"][idx_patient]["vessel"]
        if metadata is None:
            metadata = info_val["metadata"][idx_patient]["img"]
        affine, header = metadata["affine"], metadata["header"]
        nib.save(
            nib.Nifti1Image(volume_intra.astype(float), affine, header),
            os.path.join(opts.exp_dir, f"{curr_img_name}_intra.nii.gz")
        )
        if not opts.only_intra:
            nib.save(
                nib.Nifti1Image(volume_inter.astype(float), affine, header),
                os.path.join(opts.exp_dir, f"{curr_img_name}_inter.nii.gz")
            )
            nib.save(
                nib.Nifti1Image(volume_ensemble.astype(float), affine, header),
                os.path.join(opts.exp_dir, f"{curr_img_name}_pred.nii.gz")
            )
            nib.save(
                nib.Nifti1Image(volume_trans.astype(float), affine, header),
                os.path.join(opts.exp_dir, f"{curr_img_name}_trans.nii.gz")
            )
    
    os.rename("configs", "configs_train")

if __name__ == '__main__':
    main()
