from abc import abstractmethod
import torchvision.transforms as transforms
from datasets import augmentations
import torch
from skimage.transform import resize
import numpy as np
import random
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.augmentations.spatial_transformations import augment_spatial

def one_hot(seg, num_classes=2):
    return np.eye(num_classes)[np.rint(seg).astype(int)].astype(np.float32) ########### !!!!!!  ###########

########
#Resize#
########
class ResizeImg():
    def __init__(self, new_size):
        self.new_size = new_size
    def __call__(self, img):
        new_shape = (self.new_size, self.new_size)
        img = resize(img, new_shape, 3, cval=0, mode='edge', anti_aliasing=False)
        return (img, self.new_size)

class RandomResizeImg():
    def __init__(self, new_size, range_size):
        self.new_size = new_size
        self.range_size = range_size
    def __call__(self, img):
        new_size = int(random.uniform(*self.range_size) * self.new_size)
        img = resize(img, (new_size, new_size), 3, cval=0, mode='edge', anti_aliasing=False)
        return (img, new_size)

class ResizeMskTo():
    def __call__(self, sample):
        msk, new_size = sample
        new_shape = (new_size, new_size, msk.shape[-1])
        msk = resize_segmentation(msk, new_shape, order=1)
        return msk, new_size
    
class ResizeImgAndMsk():
    def __init__(self, new_size):
        self.resizeImg = ResizeImg(new_size)
        self.resizeMskTo = ResizeMskTo()
    def __call__(self, sample):
        sample["data"], new_size = self.resizeImg(sample["data"])
        sample["mask"], _ = self.resizeMskTo((sample["mask"], new_size))
        sample["weight"], _ = self.resizeMskTo((sample["weight"], new_size))
        return sample

class RandomResizeImgAndMsk:
    def __init__(self, new_size, range_size):
        self.randomResizeImg = RandomResizeImg(new_size, range_size)
        self.resizeMskTo = ResizeMskTo()
    def __call__(self, sample):
        sample["data"], new_size = self.randomResizeImg(sample["data"])
        sample["mask"], _ = self.resizeMskTo((sample["mask"], new_size))
        sample["weight"], _ = self.resizeMskTo((sample["weight"], new_size))
        return sample

class ResizeImgOnly():
    def __init__(self, new_size):
        self.resizeImg = ResizeImg(new_size)
    def __call__(self, sample):
        sample["data"], _ = self.resizeImg(sample["data"])
        return sample

#########
#Padding#
#########
class AddPadding():
    def __init__(self, output_size, pad_value, argmax=False):
        self.output_size = output_size
        self.pad_value = pad_value
        self.argmax = argmax

    def resize_image_by_padding(self, image, new_shape, pad_value):
        shape = tuple(list(image.shape))
        new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
        res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
        start = np.array(new_shape) / 2. - np.array(shape) / 2.
        res[int(start[0]) : int(start[0]) + int(shape[0]), int(start[1]) : int(start[1]) + int(shape[1])] = image
        return res
  
    def __call__(self, sample):
        sample, new_size = sample
        if sample.shape[0] > self.output_size:
            return (sample, new_size)
        if self.argmax:
            num_classes = sample.shape[-1]
            sample = np.argmax(sample, axis=-1)
        else:
            sample = sample[:,:,0]
        sample = self.resize_image_by_padding(
            sample, new_shape=[self.output_size, self.output_size], pad_value=self.pad_value
        )
        if self.argmax:
            sample = one_hot(sample, num_classes=num_classes).astype(bool)
        else:
            sample = sample[:,:,None]
        return (sample, new_size)

class AddPaddingToImgAndMsk():
    def __init__(self, output_size, pad_value_img, pad_value_msk):
        self.addPaddingToImg = AddPadding(output_size, pad_value_img)
        self.addPaddingToMsk = AddPadding(output_size, pad_value_msk, argxmax=True)
    def __call__(self, sample):
        sample["data"], _ = self.addPaddingToImg((sample["data"], None))
        sample["mask"], _ = self.addPaddingToMsk((sample["mask"], None))
        sample["weight"], _ = self.addPaddingToMsk((sample["weight"], None))
        return sample    

######
#Crop#
######
class CenterCrop():
    def __init__(self, output_size):
        self.output_size = output_size

    def center_crop_2D_image(self, img, center_crop):
        if(all(np.array(img.shape) <= center_crop)):
            return img
        center = np.array(img.shape) / 2.
        return img[
            int(center[0] - center_crop[0] / 2.) : int(center[0] + center_crop[0] / 2.),
            int(center[1] - center_crop[1] / 2.) : int(center[1] + center_crop[1] / 2.)
        ]
    
    def __call__(self, sample):
        sample, new_size = sample
        input_shape = sample.shape
        sample = self.center_crop_2D_image(sample, center_crop=[self.output_size, self.output_size, sample.shape[-1]])
        assert len(sample.shape) == len(input_shape), f"{sample.shape} and {input_shape}"
        return (sample, new_size)

class CenterCropImgAndMsk():
    def __init__(self, output_size):
        self.centerCrop = CenterCrop(output_size)
    def __call__(self, sample):
        self.addPadding.argmax = False
        sample["data"], _ = self.centerCrop((sample["data"], None))
        sample["mask"], _ = self.centerCrop((sample["mask"], None))
        sample["weight"], _ = self.centerCrop((sample["weight"], None))
        return sample 
    
########
#Mirror#
########
class MirrorTransform():
    def __call__(self,sample):
        if np.random.uniform() < 0.5:
            sample["data"] = np.copy(sample["data"][::-1])
            sample["mask"] = np.copy(sample["mask"][::-1])
            sample["weight"] = np.copy(sample["weight"][::-1])
        if np.random.uniform() < 0.5:
            sample["data"] = np.copy(sample["data"][:, ::-1])
            sample["mask"] = np.copy(sample["mask"][:, ::-1])
            sample["weight"] = np.copy(sample["weight"][::-1])
        return sample

##################
#SpatialTransform#
##################
class SpatialTransform():
    def __init__(
        self, patch_size, do_elastic_deform=False, alpha=None, sigma=None,
        do_rotation=True, angle_x=(-np.pi/6,np.pi/6), angle_y=None, angle_z=None,
        do_scale=True, scale=(0.7, 1.4), border_mode_data='constant', border_cval_data=-1, order_data=3,
        border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
        p_scale_per_sample=1, p_rot_per_sample=1,independent_scale_for_each_axis=False, p_rot_per_axis:float=1,
        p_independent_scale_per_axis: int=1
    ):
        self.params = locals()
        self.params.pop("self")
        self.params["patch_center_dist_from_border"] = list(np.array(patch_size)//2)

    def __call__(self, sample):
        sample["data"] = sample["data"][None,None,:,:,0]
        num_classes_mask = sample["mask"].shape[-1]
        sample["mask"] = np.argmax(sample["mask"], axis=-1)[None,None,:,:]
        sample["weight"] = sample["weight"][None,None,:,:,0]
        
        seed = random.randint(0, 10000)
        random.seed(seed)
        np.random.seed(seed)
        
        _, sample["mask"] = augment_spatial(sample["data"], sample["mask"], **self.params) 
        
        random.seed(seed)
        np.random.seed(seed)
        
        sample["data"], sample["weight"] = augment_spatial(sample["data"], sample["weight"], **self.params)
        
        sample["data"] = sample["data"][0,0,:,:,None]
        sample["mask"] = one_hot(sample["mask"][0,0], num_classes=num_classes_mask).astype(bool)
        sample["weight"] = sample["weight"][0,0,:,:,None]
        return sample

##########
#ToTensor#
##########
class TupleToTensor():
    def __init__(self):
        self.toTensor = transforms.ToTensor()
    def __call__(self, args):
        assert len(args) == 1 or len(args) == 2
        if len(args) > 1:
            return (self.toTensor(args[0]), args[1])
        else:
            return (self.toTensor(args[0]), None)

class DictToTensor():
    def __init__(self):
        self.toTensor = transforms.ToTensor()
    def __call__(self, sample):
        sample["data"] = self.toTensor(sample["data"])
        sample["mask"] = self.toTensor(sample["mask"])
        sample["weight"] = self.toTensor(sample["weight"])
        return sample    

class TransformsConfig(object):

    def __init__(self, opts):
        self.opts = opts

    @abstractmethod
    def get_transforms(self):
        pass

class MyInpaintingTransforms(TransformsConfig):

    def __init__(self, opts):
        super(MyInpaintingTransforms, self).__init__(opts)

    def get_transforms(self):
        transforms_dict = {
            'transform_img': transforms.Compose([
                ResizeImgAndMsk(256),
                DictToTensor()
            ]),
            'transform_msk': None,#transforms.Compose([
#                 ResizeMskTo(),
#                 TupleToTensor()
#             ]),
            'to_tensor': None,#DictToTensor(),
#             #MultiDomain
#             'transform_img_aug': transforms.Compose([
#                 RandomResizeImg(256, [0.5, 2]),
#                 AddPadding(256, -1),
#                 CenterCrop(256),
#                 TupleToTensor()
#             ]),
#             'transform_msk_aug': transforms.Compose([
#                 ResizeMskTo(),
#                 AddPadding(256, 0, argmax=True),
#                 CenterCrop(256),
#                 TupleToTensor()
#             ]),
            #MultiScanner
            'transform_img_aug': transforms.Compose([
                ResizeImgAndMsk(256),
                MirrorTransform(),
                SpatialTransform(patch_size=(256,256), angle_x=(-np.pi/6,np.pi/6), scale=(0.7,1.4), random_crop=True),
                DictToTensor()
            ]),
            'transform_msk_aug': None,
            'transform_img_val': transforms.Compose([
                ResizeImgOnly(256),
                DictToTensor()
            ]),
        }
        return transforms_dict