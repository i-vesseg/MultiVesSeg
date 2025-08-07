import os
import numpy as np
import argparse
import nibabel as nib
import torch
from types import SimpleNamespace
from torchvision import transforms, utils
from acvl_utils.morphology.morphology_helper import remove_all_but_largest_component
from tqdm import tqdm

DEBUG = False

def check_is_train():
    print(f"Current directory: {os.getcwd()}")
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

from models.psp import pSp
from models.stylegan2.model import Generator

def get_best_model(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "timestamp.txt"), "r") as file:
        timestamp = file.readlines()

    best_models = [line for line in timestamp if line.startswith("**Best saved at best_model")][-1]
    best_models = eval(best_models.split(", Loss - ")[1])

    checkpoint_path = os.path.join(checkpoint_dir, best_models[-1][0])
    return checkpoint_path

def get_best_models(checkpoint_dir):
    with open(os.path.join(checkpoint_dir, "timestamp.txt"), "r") as file:
        timestamp = file.readlines()

    best_models = [line for line in timestamp if line.startswith("**Best saved at best_model")][-1]
    best_models = eval(best_models.split(", Loss - ")[1])

    return [os.path.join(checkpoint_dir, model_name[0]) for model_name in best_models]

def interpolate(start, end, steps):
    # Ensure start and end tensors are of the same shape and type
    assert start.shape == end.shape, "Start and end tensors must have the same shape"
    assert start.dtype == end.dtype, "Start and end tensors must have the same dtype"
    
    # Calculate the step increment for each element in the tensor
    step_increment = (end - start) / (steps - 1)
    
    # Generate the sequence of interpolated tensors
    interpolated_tensors = [start + step_increment * i for i in range(0, steps)]
    
    #assert torch.all(torch.isclose(interpolated_tensors[0], start))
    #assert torch.all(torch.isclose(interpolated_tensors[-1], end))
    
    return torch.stack(interpolated_tensors, dim=0)

def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))
    return t

def one_hot(seg, num_classes=2):
    return np.eye(num_classes)[np.rint(seg).astype(int)].astype(np.float32)

def remove_all_but_largest_component_from_segmentation(msk):
    back = np.ones(msk.shape[:-1], dtype=bool)
    for label_idx in range(1, msk.shape[-1]):
        msk[..., label_idx] = remove_all_but_largest_component(msk[..., label_idx])
        back = np.logical_and(back, np.logical_not(msk[..., label_idx]))
    msk[..., 0] = back
    return msk

def generate_volume(net, args, best_models):
    z_depth = args.z_depth
    BATCH_SIZE = args.batch_size
    DEVICE = args.device
    LABEL = args.label
    n_domains = args.n_domains


    volume = np.empty([0, 512, 512])
    mask = np.empty([0, 512, 512])
    styles = torch.randn(
        1, 512, device=DEVICE
    )
    last_latent = None
    for z_axis in tqdm(range(0, z_depth, BATCH_SIZE), desc="Iterating z-axis", leave=False):
        batch_size = min(z_depth - z_axis, BATCH_SIZE)
        labels = torch.tensor([LABEL]*batch_size, device=DEVICE)
        zaxes = torch.tensor(np.arange(z_axis, z_axis + batch_size), device=DEVICE)

        if DEBUG:
            print(f"Generating {batch_size} slices with fixed labels: {labels} and spanning in z-axis: {zaxes}")
        
        clip_condition = args.clip_mean if args.clip_mean is None else args.clip_mean.repeat(batch_size, 1)
        conditions = (torch.cat([
            torch.nn.functional.one_hot(labels, num_classes=n_domains).float(),
            torch.nn.functional.one_hot(zaxes, num_classes=29).float()
        ], dim=-1), clip_condition)

        imgs, latents = net.decoder([styles.repeat(batch_size, 1)], conditions, return_latents=True)
        imgs = (norm_range(imgs, (-1, 1)) * 2) - 1
        if DEBUG:
            utils.save_image(
                imgs,
                f"img1.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )

            print(f"In return we got images of shape {imgs.shape} and latents of shape {latents.shape}")

        if last_latent is not None:
            latents = torch.cat([last_latent, latents], axis=0)
        
        for i in tqdm(range(len(latents)), desc="Iterating latents", leave=False):
            if i < len(latents) - 1:
                inter_latents = interpolate(latents[i], latents[i+1], args.slices_in_one_z + 1)[:-1]
                if i == len(latents)-2:
                    inter_latents = inter_latents[:-1]
            else:
                inter_latents = latents[i:]

            if DEBUG:
                print(f"Now we generate {len(inter_latents)} images from latent {i} to latent {i+1}")

            for latents_count in tqdm(range(0, len(inter_latents), BATCH_SIZE), desc="Iterating inter_latents", leave=False):
                batch_size = min(len(inter_latents) - latents_count, BATCH_SIZE)
                
                inter_imgs, _ = net.decoder.g_synthesis(inter_latents[latents_count: latents_count + batch_size])
                inter_imgs = (norm_range(inter_imgs, (-1, 1)) * 2) - 1
                
                if DEBUG:
                    utils.save_image(
                        inter_imgs,
                        f"img2.png",
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )
                
                """print(torch.unique(inter_imgs))
                inter_imgs = torch.tensor(np.load(
                    "/home/galati/preprocessing/preprocess_MMs_B/train/A1D0Q7_ed_slice005.npy", allow_pickle=True
                ).item()["data"].transpose(2,0,1)[None,...]).cuda()
                utils.save_image(
                    inter_imgs,
                    f"img4.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
                print(torch.unique(inter_imgs))"""
                
                inter_msks = 0
                for best_model_id, best_model_path in enumerate(best_models[::-1]):
                    if best_model_id == 0:
                        assert args.ckpt == best_model_path
                    net.opts.checkpoint_path = best_model_path
                    del net.latent_avg
                    net.load_weights(verbose=False)
                    inter_msk, _ = net(
                        net.face_pool(inter_imgs), torch.tensor([[1, 0]]).repeat(len(inter_imgs), 1).float().cuda(),
                        resize=False, return_latents=True, feature_scale=args.feature_scale
                    )
                    _, inter_msk = inter_msk[:,:1], inter_msk[:,1:]
                    inter_msks += inter_msk
                inter_msks /= len(best_models)
                
                if DEBUG:
                    utils.save_image(
                        (norm_range(_, (-1, 1)) * 2) - 1,
                        f"img3.png",
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                    )

                volume = np.concatenate([volume, inter_imgs[:, 0].cpu().numpy()], axis=0)
                mask = np.concatenate([mask, torch.argmax(inter_msks, dim=1).cpu().numpy()])

            if DEBUG:
                print(f"After concatenating the results, volume has shape {volume.shape} and mask has shape {mask.shape}")
        last_latent = latents[-1:]

    return volume, mask
                 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process input parameters.')
    # Compulsory arguments
    parser.add_argument("--ckpt_dir", type=str, required=True, help="The checkpoint @ stage 2.1")
    parser.add_argument('--label', type=int, required=True, help='Label (compulsory)')
    parser.add_argument('--n_volumes', type=int, required=True, help='Number of volumes (compulsory)')
    parser.add_argument('--slices_in_one_z', required=True, type=int)
    parser.add_argument('--n_segmentation_labels', required=True, type=int)
    parser.add_argument('--out_dir', required=True, type=str)
    
    # Optional arguments with default values
    parser.add_argument('--clip_mean', type=str, default=None)
    parser.add_argument('--z_depth', type=int, default=29, help='Depth of Z (default: 29)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: "cuda")')
    parser.add_argument('--n_domains', type=int, default=3, help='Number of domains (default: 3)')
    parser.add_argument('--remove_largest_components', action="store_true", help='Remove the largest components from generated masks')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    
    # Use the arguments
    print(f"ckpt: {args.ckpt_dir}")
    print(f"z_depth: {args.z_depth}")
    print(f"batch_size: {args.batch_size}")
    print(f"device: {args.device}")
    print(f"label: {args.label}")
    print(f"n_domains: {args.n_domains}")
    print(f"n_volumes: {args.n_volumes}")
    print(f"clip_mean: {args.clip_mean}")
    
    best_models = get_best_models(args.ckpt_dir)
    args.ckpt = best_models[-1]
    ckpt = torch.load(args.ckpt)
    args.feature_scale = min(1.0, 0.0001*ckpt["global_step"])
    
    opts = SimpleNamespace(**ckpt["opts"])
    opts.checkpoint_path = args.ckpt
    opts.n_domains = args.n_domains
    opts.src_tgt_domains = [args.label]
    if "arcface_model" not in opts.__dict__:
        opts.arcface_model = "./pretrained_models/backbone.pth"

    net = pSp(opts)
    net = net.cuda().eval()

    if args.clip_mean is not None:
        args.clip_mean = torch.tensor(np.load(args.clip_mean), device=args.device).float()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    
    with torch.no_grad():
        for volume_id in tqdm(range(args.n_volumes), desc="Generating volumes"):
            volume, mask = generate_volume(net, args, best_models)
            mask = one_hot(mask, args.n_segmentation_labels).astype(bool)
            if args.remove_largest_components: 
                for i in range(len(mask)):
                    mask[i] = remove_all_but_largest_component_from_segmentation(mask[i])
            if DEBUG:
                nib.save(nib.Nifti1Image(volume, None), "img.nii.gz")
                nib.save(nib.Nifti1Image(np.argmax(mask, axis=-1).astype(np.float32), None), "msk.nii.gz")
                raise Exception("CHECK")
            for slice_id, (img_slice, msk_slice) in enumerate(zip(volume, mask)):
                slice = {
                    "data": img_slice[:,:,None].astype(np.float32),
                    "mask": msk_slice,
                    "weight": None,
                }
                np.save(os.path.join(args.out_dir,"fakeVolume{}_slice{:03d}.npy".format(volume_id, slice_id)), slice)
    
    os.rename("configs", "configs_train")

if __name__ == "__main__":
    main()