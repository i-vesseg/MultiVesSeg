import os
import nibabel as nib
import numpy as np
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import copy
import skimage
import re
import pickle

from PIL import Image
from tempfile import NamedTemporaryFile as NTF
from tqdm import tqdm
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation

from nilearn.image import resample_img

do_flip=False
def get_fdata(self):
    global do_flip
    img = nib_get_fdata(self)
    if do_flip:
        img = img[:,::-1]
    return img
nib_get_fdata = nib.nifti1.Nifti1Image.get_fdata
nib.nifti1.Nifti1Image.get_fdata = get_fdata

img_path_to_img_id = None
TRANSPOSE = [2, 0, 1]

reference_affine = None
def my_load(nib_path, type="data"):
    global reference_affine
    img = nib_load(nib_path)
    if reference_affine is not None:
        img = resample_img(img, reference_affine, interpolation="nearest" if type=="mask" else "continuous")
    return img
nib_load = nib.load
nib.load = my_load

###################
###Extract paths###
###################

def load_info_from_checkpoint(info_path):
    with open(info_path, "rb") as file:
        info_dict = pickle.load(file)
    before_split = info_dict["val"] is None and info_dict["test"] is None
    if before_split:
        info = {"train": extract_paths(), "val": None, "test": None}
    else:
        info = {
            "train": extract_info_and_masks(),
            "val": extract_info_and_masks(),
            "test": extract_info_and_masks(),
        }
    for phase in ["train"] if before_split else ["train", "val", "test"]:
        for k, v in info_dict[phase].items():
            setattr(info[phase], k, v)
    return info

def save_info_to_checkpoint(info, info_path):
    info_dict = {}
    for k in info:
        info_dict[k] = {**info[k].__dict__} if info[k] is not None else None
        if info_dict[k] is not None and "shapesAfterCropping" not in info_dict[k] and hasattr(info[k], "shapesAfterCropping"):
            info_dict[k]["shapesAfterCropping"] = info[k].shapesAfterCropping()  
    with open(info_path, "wb") as file:
        pickle.dump(info_dict, file)

####################
##Display examples##
####################

def numpy_to_PIL(img, min_val=None, max_val=None, mask=None):
    if min_val is None:
        min_val = np.min(img)
    if max_val is None:
        max_val = np.max(img)
    img = Image.fromarray(
        ( (img - min_val) / (max_val - min_val) *255 ).astype(np.uint8)
    ).convert("RGBA")
    
    if mask is not None:
        bkg = Image.blend(Image.new("RGBA", img.size, (255, 0, 0, 255)), img, 0.5)
        mask = np.stack([mask, mask, mask, mask], axis=-1)
        #img = Image.fromarray(np.where(mask==0, np.array(img), np.array(bkg)))
        img = Image.fromarray(((1-mask)*np.array(img) + mask*np.array(bkg)).astype(np.uint8))
    
    return img

def display_slice(img, min_val=None, max_val=None, mask=None):
    img = numpy_to_PIL(img, min_val, max_val, mask)
    display(img.rotate(90, expand=True))

def display_middle_slice(img, axis=2, mask=None, downsample_factor=1):
    if not isinstance(axis, list):
        axes = [axis]
    else:
        axes = axis
    img = img.transpose(1, 2, 0)
    if mask is not None:
        mask = mask.transpose(1, 2, 0)
    for axis in axes:
        tr_img = img.transpose([axis, *[i for i in range(3) if i!= axis]])
        slice = tr_img[tr_img.shape[0]//2]
        if mask is not None:
            tr_mask = mask.transpose([axis, *[i for i in range(3) if i!= axis]])
            slice_mask = tr_mask[tr_mask.shape[0]//2]
        else:
            slice_mask = None
        slice = numpy_to_PIL(slice, np.min(img), np.max(img), slice_mask)
        slice = slice.rotate(90, expand=True)
        display(slice.resize(tuple(np.array(slice.size)//downsample_factor)))

def load_volume(img_path, mask_path=None, mask=None, display_header=False):
    img = nib.load(img_path)
    if display_header:
        print(img.header)
    img = img.get_fdata()
    img = img.transpose(*TRANSPOSE)
    if np.any(np.isnan(img)):
        img = np.where(np.isnan(img), 0, img)
    if mask_path is not None:
        mask = nib.load(mask_path, type="mask").get_fdata()
    if mask is not None:
        mask = mask.transpose(*TRANSPOSE)
    return img, mask

def load_and_display_middle_slice(img_path, axis=2, mask_path=None, mask=None, display_header=False):
    img, mask = load_volume(img_path, mask_path, mask, display_header)
    display_middle_slice(img, axis, mask)
    
def print_bold(x):
    print("\033[1m{}\033[0m".format(x))

def is_medical_volume(path):
    return path is not None and os.path.isfile(path) and (path.endswith(".nii.gz") or path.endswith(".nii"))

###############################
##Extract spacings and shapes##
###############################

def open_volume(img_path):#, path_must_contain):
    #if path_must_contain not in img_path or "TOF_echo" in img_path or img_path.endswith("_bet.nii.gz") or img_path.endswith("_bet_mask.nii.gz"):
    #    raise Exception("Wrong path")
    img = nib.load(img_path)
    metadata = {
        "header": img.header,
        "affine": img.affine
    }
    spacing = img.header["pixdim"][[1,2,3]]
    img = img.get_fdata()
    shape = img.shape
    if len(shape) != 3:
        raise Exception("Wrong shape")
    return img, metadata, spacing, shape

def cropVolume(volume):
    crop = []
    axes = np.arange(len(volume.shape))
    for axis in axes:
        other_axes = tuple(np.delete(axes, axis).tolist())
        
        intensity_max = np.max(volume, axis=other_axes)
        cumsum = np.cumsum(intensity_max).astype(intensity_max.dtype)
        #plt.hist(cumsum, bins=255)
        #plt.show()
        cumsum_counts, cumsum_bins = np.histogram(cumsum, 255)
        
        try:
            thr = skimage.filters.threshold_minimum(cumsum_counts)
        except Exception:
            crop += [slice(0, volume.shape[axis])]
            continue
        
        start, end = np.argmin(cumsum_counts > thr), -1*np.argmin(cumsum_counts[::1] > thr)
        start, end = cumsum_bins[start], cumsum_bins[end]
        start, end = np.argmin(cumsum < start), np.argmin(cumsum < end)
        crop += [slice(start, end if end!=0 else volume.shape[axis])]
    #print(crop)
    return tuple(crop)

"""def get_slice_size(slc):
    return len(range(*slc.indices(slc.stop)))

def cropToShape(crop):
    return [get_slice_size(c) for c in crop]"""

def cropToShape(crop, shape_before):
    volume = np.zeros(shape_before, dtype=bool)
    shape_after = volume[crop].shape
    return shape_after

class extract_paths:
    def __init__(self, path_rule=""):
        self.path_rule = path_rule
        self.paths = []
    def __call__(self, img_path):
        if not re.match(self.path_rule, os.path.basename(img_path)):
            return
        self.paths += [img_path]

class extract_info(extract_paths):
    def __init__(self, path_rule=""):
        super().__init__(path_rule)
        self.metadata = []
        self.spacings = []
        self.shapes = []
        self.crops = []
    def __call__(self, img_path):
        len_paths = len(self.paths)
        super().__call__(img_path)
        if len(self.paths) == len_paths:
            raise Exception("Wrong path")
        
        img, metadata, spacing, shape = open_volume(img_path)
        crop = cropVolume(img)
        
        self.metadata += [metadata]
        self.spacings += [spacing]
        self.shapes += [shape]
        self.crops += [crop]
    def shapesAfterCropping(self):
        return [cropToShape(crop, shape) for crop, shape in zip(self.crops, self.shapes)]
    
class extract_info_and_masks(extract_info):
    def __init__(self, path_rule=""):
        super().__init__(path_rule)
        self.brain_paths = []
        self.weight_paths = []
        self.vessel_paths = []
    def __call__(self, img_path):
        try:
            super().__call__(img_path)
        except Exception as e:
            if not str(e).startswith("Wrong "):
                print(e, ":", img_path)
            return
        img_path, metadata = self.paths.pop(), self.metadata.pop()
        
        brain_path = img_path_to_brain_path(img_path)
        weight_path = img_path_to_weight_path(img_path)
        vessel_path = img_path_to_vessel_path(img_path)
        
        brain_mask = nib.load(brain_path, type="mask") if is_medical_volume(brain_path) else None
        vessel_mask = nib.load(vessel_path, type="mask") if is_medical_volume(vessel_path) else None
        
        metadata = {
            "img": metadata,
            "brain": {"header": brain_mask.header, "affine": brain_mask.affine} if brain_mask is not None else None,
            "vessel": {"header": vessel_mask.header, "affine": vessel_mask.affine} if vessel_mask is not None else None,
        }
        
        self.paths += [img_path]
        self.brain_paths += [brain_path if is_medical_volume(brain_path) else None]
        self.weight_paths += [weight_path if weight_path is not None and os.path.isfile(weight_path) else None]
        self.vessel_paths += [vessel_path if is_medical_volume(vessel_path) else None]
        self.metadata += [metadata]
        
def search_nii(path, extraction):
    #if "main_folder" not in extraction.__dict__:
    #    extraction.main_folder = path
    if os.path.isfile(path):
        if path.endswith(".nii.gz") or path.endswith(".nii"):
            extraction(path)
    else:
        for subpath in sorted(os.listdir(path)):
            subpath = os.path.join(path, subpath)
            search_nii(subpath, extraction)

def loop_nii(paths, extraction):
    for path in paths:
        extraction(path)

def eval_statistics(s):
    return {
        k: np.percentile(np.array(s), q, axis=0)
    for k, q in zip(["min", "median", "max"], [0, 50, 100])}

def display_info(extract_info_train, extract_info_val):
    len_train = len(extract_info_train.shapes)
    len_val = len(extract_info_val.shapes)

    slides_train = sum([shape[2] for shape in extract_info_train.shapesAfterCropping()])
    slides_val = sum([shape[2] for shape in extract_info_val.shapesAfterCropping()])

    spacing_train = eval_statistics(extract_info_train.spacings)
    spacing_val = eval_statistics(extract_info_val.spacings)
    
    shape_train = eval_statistics(extract_info_train.shapesAfterCropping())
    shape_val = eval_statistics(extract_info_val.shapesAfterCropping())

    with np.printoptions(formatter={'float': "{:06.2f}".format}):#precision=2, suppress=True):
        info_df = pd.DataFrame.from_dict({
                "TRAIN": [
                    len_train, slides_train,
                    str(spacing_train["median"]),
                    "{} - {}".format(spacing_train["min"], spacing_train["max"]),
                    str(shape_train["median"]),
                    "{} - {}".format(shape_train["min"], shape_train["max"]),
                ],
                "VAL": [
                    len_val, slides_val,
                    str(spacing_val["median"]),
                    "{} - {}".format(spacing_val["min"], spacing_val["max"]),
                    str(shape_val["median"]),
                    "{} - {}".format(shape_val["min"], shape_val["max"]),
                ]
            }, 
            columns=["#volumes", "#slices", "spacing median", "spacing range", "shape median", "shape range"],
            orient="index"
        )
    
    with np.printoptions(precision=3, suppress=True):
        display(info_df)

def chop_mask(src_path, dst_path):
    mask = nib.load(src_path, type="mask")
    mask, affine, header = mask.get_fdata(), mask.affine, mask.header
    chopped_mask = np.zeros(mask.shape)
    z_size = mask.shape[-1] * 2 // 3
    chopped_mask[:, :, :z_size] = mask[:, :, :z_size]
    nib.save(nib.Nifti1Image(chopped_mask, affine, header), dst_path)
    return

def get_target_spacing(spacings, shapes):
    spacings = np.asarray(spacings)
    shapes = np.asarray(shapes)
    
    spacing_median = np.percentile(spacings, 50, axis=0)    
    shapes_new = np.round(spacings / spacing_median * shapes).astype(int)
    shape_max = np.max(shapes_new, axis=0)
    
    spacing_target = np.where(shape_max > 512, spacing_median * shape_max / 512, spacing_median)
    
    #CHECK
    #shapes_new = np.round(spacings / spacing_target * shapes).astype(int)
    #raise Exception(shapes_new)
    return spacing_target
        
####################################
##########Standardization###########
####################################

def resizeVolume(image, spacing_target, current_spacing, ignore_z=False):
    if ignore_z:
        spacing_target[-1] = current_spacing[-1]
    new_shape = np.round(current_spacing / spacing_target * image.shape).astype(int)
    if all(new_shape == image.shape):
        return image
    image = resize(image, new_shape, 3, cval=0, mode='edge', anti_aliasing=False)
    return image

def standardizeVolume(image, mean=None, std=None):
    if mean is None or std is None:
        mean = np.mean(image)
        std = np.std(image)
    image -= mean
    image /= std + 1e-8
    return image

def split_path(path):
    tail, head = os.path.split(path)
    if tail == path:
        return [tail]
    return [*split_path(tail), head]

def load_weight(weight_path, img_shape):
    #This function is conceived to work only with the TopCow dataset
    with open(weight_path) as file:
        crop_info = file.readlines()

    size = [int(i) for i in crop_info[1].split(": ")[1].split()]
    location = [int(i) for i in crop_info[2].split(": ")[1].split()]
    crop = tuple([slice(l, l + s) for l,s in zip(location, size)])
    
    weight_mask = np.zeros(img_shape)
    weight_mask[crop] = 1
    
    global do_flip
    if do_flip:
        weight_mask = weight_mask[:,::-1]
        start_invert = weight_mask.shape[1] - crop[1].stop
        stop_invert = weight_mask.shape[1] - crop[1].start
        crop = (crop[0], slice(start_invert, stop_invert), crop[2])
    
    return weight_mask, crop

def preprocessing_loop(info, out_dir, target_spacing=None, discard_extracerebral_slices=True, ignore_z=False, dont_standardize=False):
    info.preprocessed_paths = []
    info.min_values = []
    info.max_values = []
    info.depths = []
    info.z_splits = []
    info.areas = []
    info.shapesBeforePadding = []
    counter = 0
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    for i, img_path in enumerate(tqdm(info.paths)):
        brain_path, weight_path, vessel_path, crop = info.brain_paths[i], info.weight_paths[i], info.vessel_paths[i], info.crops[i]

        #Open image and masks
        img = nib.load(img_path).get_fdata()
        
        if brain_path is not None:
            brain_mask = nib.load(brain_path, type="mask").get_fdata().squeeze()
            assert brain_mask.shape == img.shape
        else:
            brain_mask = np.zeros(img.shape)
            
        if weight_path is not None:
            weight_mask, _ = load_weight(weight_path, img.shape)
            assert weight_mask.shape == img.shape
        else:
            weight_mask = np.ones(img.shape)

        if vessel_path is not None:
            vessel_mask = nib.load(vessel_path, type="mask").get_fdata().squeeze()
            assert vessel_mask.shape == img.shape
        else:
            vessel_mask = np.zeros(img.shape)

        #Crop volumes
        img = img[crop]
        brain_mask = brain_mask[crop]
        weight_mask = weight_mask[crop]
        vessel_mask = vessel_mask[crop]

        #Deal with metadata
        metadata = copy.deepcopy(info.metadata[i])
        if metadata["brain"] is not None:
            metadata["brain"]["header"].set_data_dtype("float32")
            if metadata["vessel"] is not None:
                metadata["vessel"]["affine"] = copy.deepcopy(metadata["brain"]["affine"])

        #print(img.shape)
        #display_middle_slice(img, axis=0),#[:, img.shape[1]//2, :]),
        #display_middle_slice(brain_mask, axis=0),#[:, img.shape[1]//2, :]),
        #display_middle_slice(vessel_mask, axis=0)#[:, img.shape[1]//2, :])
        #raise Exception("CHECK")
        
        #Resizing
        if target_spacing is not None:
            spacing = info.spacings[i]
            img = resizeVolume(img, target_spacing, spacing, ignore_z=ignore_z)
            #assert np.all(np.asarray(img.shape[:2]) <= 512), "Maximum size must be equal to 512"
            brain_mask = resize_segmentation(brain_mask, img.shape, order=1)
            weight_mask = resize_segmentation(weight_mask, img.shape, order=1)
            vessel_mask = resize_segmentation(vessel_mask, img.shape, order=1)
        
        #print(img.shape)
        #display_middle_slice(img, axis=0),#[:, img.shape[1]//2, :]),
        #display_middle_slice(brain_mask, axis=0),#[:, img.shape[1]//2, :]),
        #display_middle_slice(vessel_mask, axis=0)#[:, img.shape[1]//2, :])
        #raise Exception("CHECK")

        #Transpose so that we start treating volumes as a list of slices
        img = img.transpose(*TRANSPOSE)
        brain_mask = brain_mask.transpose(*TRANSPOSE)
        weight_mask = weight_mask.transpose(*TRANSPOSE)
        vessel_mask = vessel_mask.transpose(*TRANSPOSE)

        #In case the brain mask is available, you can discard useless slices
        z_mask = np.where(brain_mask)[0]
        if not discard_extracerebral_slices or len(z_mask) == 0:
            first_slice, last_slice = 0, len(brain_mask)
        else:
            first_slice, last_slice = np.min(z_mask), np.max(z_mask) - len(brain_mask)
            if last_slice >= 0:
                last_slice = len(brain_mask)
        
        img = img[first_slice:last_slice]
        brain_mask = brain_mask[first_slice:last_slice]
        weight_mask = weight_mask[first_slice:last_slice]
        vessel_mask = vessel_mask[first_slice:last_slice]
        depth = len(brain_mask)
        
        if depth <= 0:
            print("Segmentation failed")
            return
        
        #display_middle_slice(img, axis=0),#[:, img.shape[1]//2, :]),
        #display_middle_slice(brain_mask, axis=0),#[:, img.shape[1]//2, :]),
        #display_middle_slice(vessel_mask, axis=0)#[:, img.shape[1]//2, :])
        #raise Exception("CHECK")

        #Standardization
        if not dont_standardize:
            img = standardizeVolume(img)

        #raise Exception(
        #    nib.save(nib.Nifti1Image(img, None), "test.nii.gz"),
        #    nib.save(nib.Nifti1Image(brain_mask.astype(np.float32), None), "test_brain_mask.nii.gz"),
        #    nib.save(nib.Nifti1Image(vessel_mask.astype(np.float32), None), "test_vessel_mask.nii.gz")
        #)

        #Save preprocessed volumes and info
        if img_path_to_img_id is not None:
            out_path = img_path_to_img_id(img_path) + ".npy"
        else:
            out_path = os.path.split(img_path)[-1].replace(".nii.gz", ".npy")
        out_path = os.path.join(out_dir, out_path)
        
        info.min_values += [np.min(img)]
        info.max_values += [np.max(img)]
        info.depths += [depth]
        info.z_splits += [(first_slice, last_slice)]
        info.areas += [(np.sum(brain_mask[:5]), np.sum(brain_mask[-5:]))]
        info.preprocessed_paths += [out_path]
        info.shapesBeforePadding += [img.shape]

        np.save(out_path, {
            "data": img.astype(np.float32),
            "brain_mask": brain_mask.astype(bool),
            "weight_mask": weight_mask.astype(bool),
            "vessel_mask": vessel_mask.astype(bool if len(np.unique(vessel_mask)) == 2 else int),
        })
        
        counter += len(img)
        print("\rSlices processed: {}".format(counter))
    
##########################################    
##Outliers removal and slices extraction##
##########################################

def extraction_loop(info, out_dir, outliers=[], depth=None, delete_old=True):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    info.slices_paths = []
    i = 0
    for volume_path in tqdm(info.preprocessed_paths[:]):
        if volume_path in outliers:
            info.paths.pop(i)
            info.metadata.pop(i)
            info.spacings.pop(i)
            info.shapes.pop(i)
            info.brain_paths.pop(i)
            info.vessel_paths.pop(i)
            info.preprocessed_paths.pop(i)
            info.min_values.pop(i)
            info.max_values.pop(i)
            info.depths.pop(i)
            info.z_splits.pop(i)
            info.areas.pop(i)
            info.shapesBeforePadding.pop(i)
            continue
        
        volume = np.load(volume_path, allow_pickle=True).item()
        
        if depth is not None:
            new_shape = [depth, *volume["data"].shape[1:]]
            volume["data"] = resize(volume["data"], new_shape, 3, cval=0, mode='edge', anti_aliasing=False)
            volume["brain_mask"] = resize_segmentation(volume["brain_mask"], new_shape, order=1)
            volume["weight_mask"] = resize_segmentation(volume["weight_mask"], new_shape, order=1)
            volume["vessel_mask"] = resize_segmentation(volume["vessel_mask"], new_shape, order=1)
            start = 0
        else:
            start = 0#info.depths[i][0]
        
        slice_path = os.path.join(out_dir, os.path.basename(volume_path).replace(".npy", "_slice{:03d}.npy"))
        info.slices_paths += [[]]
        for slice_id, (slice, mask, weight, label) in enumerate(zip(
            volume["data"], volume["brain_mask"], volume["weight_mask"], volume["vessel_mask"]
        ), start=start):
            np.save(slice_path.format(slice_id), {
                "data": slice, "brain_mask": mask, "weight_mask": weight, "vessel_mask": label
            })
            info.slices_paths[i] += [slice_path.format(slice_id)]
        i += 1
        if delete_old:
            os.remove(volume_path)

#############################
##Intensity Values Analysis##
#############################

def intensity_loop(info, min_value, max_value, n_bins=256):
    info.bins = np.linspace(min_value, max_value, n_bins + 1)
    #info.bins = np.insert(info.bins, 1, info.bins[0] + np.finfo(info.bins.dtype).eps)
    info.intensity_count = np.zeros(n_bins)
    
    for slice_paths in tqdm(info.slices_paths):
        for slice_path in slice_paths:
            slice = np.load(slice_path, allow_pickle=True).item()
            hist = np.histogram(slice["data"], bins=info.bins)
            info.intensity_count += hist[0]
    
    info.intensity_perc = info.intensity_count / sum(info.intensity_count) * 100
    info.intensity_perc = np.cumsum(info.intensity_perc)

def display_intensity(info):
    plt.bar(info.bins[:-1], info.intensity_count, color='r', alpha=0.7, width=0.1)
    plt.show()
    plt.plot(info.bins[:-1], info.intensity_perc)
    plt.show()

def get_percentile(bins, intensity_count, p):
    intensity_perc = intensity_count / sum(intensity_count) * 100
    intensity_perc = np.cumsum(intensity_perc)
    bins = np.array([(s, e) for s, e in zip(bins[:-1], bins[1:])])
    index = np.where(intensity_perc >= p)[0][0]
    return bins[index][1]

def display_clipped_slice(img_path, thr_low, thr_up):
    img = np.load(img_path, allow_pickle=True).item()["data"]
    display_slice(img)
    
    display_slice(np.clip(img, thr_low, thr_up), np.min(img), np.max(img))
    
    mask_low = img >= thr_low
    mask_low = np.stack([mask_low]*3, axis=2)
    mask_up = img <= thr_up
    mask_up = np.stack([mask_up]*3, axis=2)
    
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) *255
    RGB_img = np.stack([img, img, img], axis=2)
    R___img = np.stack([img, np.zeros(img.shape), np.zeros(img.shape)], axis=2)
    _GB_img = 255 - np.stack([np.full(img.shape, 255), img, img], axis=2)
    img = np.where(mask_low, RGB_img, _GB_img)
    img = np.where(mask_up, img, R___img)
    #display(Image.fromarray(img.astype(np.uint8)))
    display_slice(img)

#######################################################
##Normalization, Padding & Cropping, One-Hot Encoding##
#######################################################
    
def normalizeSlice(image, min=None, max=None):
    if max is None or min is None:
        max = np.max(image)
        min = np.min(image)
    image -= min
    image *= 2
    image /= max - min
    image -= 1
    return image

def padding(img, new_shape, pad_value = 0):
    shape = tuple(list(img.shape))
    new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
    res = np.ones(list(new_shape), dtype=img.dtype) * pad_value
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    res[int(start[0]) : int(start[0]) + int(shape[0]), int(start[1]) : int(start[1]) + int(shape[1])] = img
    return res
    
def center_crop(slice, center_crop):
    if(all(np.array(slice.shape) <= center_crop)):
        return slice
    center = np.array(slice.shape) / 2.
    return slice[int(center[0] - center_crop[0] / 2.) : int(center[0] + center_crop[0] / 2.), int(center[1] - center_crop[1] / 2.) : int(center[1] + center_crop[1] / 2.)]

def one_hot(seg, num_classes=2):
    return np.eye(num_classes)[np.rint(seg).astype(int)].astype(np.float32) ########### !!!!!!  ###########

def add_dim(img):
    new_img = img[:, :, np.newaxis] 
    #print(f'Old Shape: {img.shape} ==> New Shape: {new_img.shape}')
    return new_img

def preprocessSlice(img, type="data", size=512, num_classes=2, pad_value=-1):
    img = padding(img, (size,size), pad_value=-1 if type=="data" else pad_value)
    img = center_crop(img, (size,size))
    if type == "mask":
        img = one_hot(img, num_classes=num_classes) #--> (512,512,2) 
    else:
        img = add_dim(img) #--> (512,512,1)
    return img

def normalization_loop(info, out_dir, min_value, max_value, delete_old=True, size=512, num_classes=2, discard_brain=False):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for slice_paths in tqdm(info.slices_paths):
        for slice_path in slice_paths:
            slice_id = os.path.basename(slice_path)
            slice = np.load(slice_path, allow_pickle=True).item()
            slice["data"] = np.clip(slice["data"], min_value, max_value)
            slice["data"] = normalizeSlice(slice["data"], min_value, max_value)
            slice["data"] = preprocessSlice(slice["data"], size=size)
            slice["brain_mask"] = preprocessSlice(slice["brain_mask"], type="mask", size=size, pad_value=0)[:,:,1:]
            if np.all(slice["weight_mask"]):
                slice["weight_mask"] = preprocessSlice(slice["weight_mask"], type="mask", size=size, pad_value=1)[:,:,1:]
            else:
                slice["weight_mask"] = preprocessSlice(slice["weight_mask"], type="mask", size=size, pad_value=0)[:,:,1:]
            slice["vessel_mask"] = preprocessSlice(
                slice["vessel_mask"], type="mask", size=size, num_classes=num_classes, pad_value=0
            )[:,:,1:]

            slice["vessel_mask"] = np.where(slice["vessel_mask"] == 1, 1., 0.)
            slice["brain_mask"] = np.where((slice["brain_mask"] == 1) & np.all(slice["vessel_mask"] == 0, axis=-1, keepdims=True), 1., 0.)
            slice["background"] = np.where((slice["brain_mask"] == 0) & np.all(slice["vessel_mask"] == 0, axis=-1, keepdims=True), 1., 0.)
            
            if not discard_brain:
                slice["mask"] = np.concatenate([slice["background"], slice["brain_mask"], slice["vessel_mask"]], axis=-1)
            else:
                slice["mask"] = np.concatenate([slice["background"], slice["vessel_mask"]], axis=-1)
            
            slice = {
                "data": slice["data"].astype(np.float32),
                "mask": slice["mask"].astype(bool),
                "weight": slice["weight_mask"].astype(bool) if np.any(slice["weight_mask"] == 0) else None
            }
            
            
            np.save(os.path.join(out_dir,"{}".format(slice_id)), slice)
            if delete_old:
                os.remove(slice_path)