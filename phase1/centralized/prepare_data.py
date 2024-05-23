import argparse
from io import BytesIO
import multiprocessing
from functools import partial
import os
import numpy as np

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def resize_and_convert(img, size, resample, quality=100):
    img = trans_fn.resize(img, size, resample)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format="jpeg", quality=quality)
    val = buffer.getvalue()

    return val


def resize_multiple(
    img, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS, quality=100
):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, resample, quality))

    return imgs


def resize_worker(img_file, sizes, resample):
    i, file = img_file
    #img = Image.open(file)
    #img = img.convert("RGB")
    #out = resize_multiple(img, sizes=sizes, resample=resample)
    out = np.load(file, allow_pickle=True).item()["data"][None,...]
    return i, out


def prepare(
    env, dataset, n_worker, sizes=(128, 256, 512, 1024), resample=Image.LANCZOS
):
    resize_fn = partial(resize_worker, sizes=sizes, resample=resample)

    #files = sorted(dataset.imgs, key=lambda x: x[0])
    #files = [(i, file) for i, (file, label) in enumerate(files)]
    files = sorted(dataset, key=lambda x: x[0])
    
    #ADD Z AXIS
    
    labels = {i: 0 if "acq-TOF" in file else 1 for i, file in enumerate(files)}
    np.save(env.path()+"/labels", labels)
    
    zaxes = {i: file.split("_slice")[1].split(".npy")[0] for i, file in enumerate(files)}
    
    depths = {}
    for i, file in enumerate(files):
        key = file.split("_slice")[0]
        depths[key] = 1 if key not in depths else depths[key]+1        
    depths = {i: depths[file.split("_slice")[0]] for i, file in enumerate(files)}
    
    zaxes = {i: "{:02d}".format(int(zaxes[i]) * 29 // depths[i]) for i in zaxes.keys()}
    np.save(env.path()+"/zaxes", zaxes)
    
    files = [(i, file) for i, file in enumerate(files)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f"{size}-{str(i).zfill(5)}-{labels[i]}-{zaxes[i]}".encode("utf-8")

                with env.begin(write=True) as txn:
                    txn.put(key, img)

            total += 1

        with env.begin(write=True) as txn:
            txn.put("length".encode("utf-8"), str(total).encode("utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for model training")
    parser.add_argument("--out", type=str, help="filename of the result lmdb dataset")
    parser.add_argument(
        "--size",
        type=str,
        default="128,256,512,1024",
        help="resolutions of images for the dataset",
    )
    parser.add_argument(
        "--n_worker",
        type=int,
        default=8,
        help="number of workers for preparing dataset",
    )
    parser.add_argument(
        "--resample",
        type=str,
        default="lanczos",
        help="resampling methods for resizing images",
    )
    parser.add_argument("--src_path", type=str, help="path to the source image dataset")
    parser.add_argument("--tgt_path", type=str, help="path to the target image dataset")

    args = parser.parse_args()

    resample_map = {"lanczos": Image.LANCZOS, "bilinear": Image.BILINEAR}
    resample = resample_map[args.resample]

    sizes = [int(s.strip()) for s in args.size.split(",")]

    print(f"Make dataset of image sizes:", ", ".join(str(s) for s in sizes))

    imgset = [
        os.path.join(args.src_path, file_path) for file_path in os.listdir(args.src_path) if file_path.endswith(".npy")
    ] + [
        os.path.join(args.tgt_path, file_path) for file_path in os.listdir(args.tgt_path) if file_path.endswith(".npy")
    ]#datasets.ImageFolder(args.path)

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        prepare(env, imgset, args.n_worker, sizes=sizes, resample=resample)
