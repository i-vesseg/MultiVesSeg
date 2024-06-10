"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
    '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    if os.path.isdir(dir):
        images = [os.path.join(dir, fname) for fname in os.listdir(dir) if is_image_file(fname)]
    else:
        images = [path for path in glob.glob(dir + "*") if is_image_file(os.path.basename(path))]
    assert all([os.path.isfile(img) for img in images]), '%s is not a valid path' % dir
    assert len(images) > 0, '%s is not a valid path' % dir
    return images
