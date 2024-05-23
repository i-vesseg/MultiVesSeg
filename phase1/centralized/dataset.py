from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        self.labels = np.load(path+"/labels.npy", allow_pickle=True).item()
        self.zaxes = np.load(path+"/zaxes.npy", allow_pickle=True).item()

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            label = self.labels[index]
            zaxis = self.zaxes[index]
            key = f'{self.resolution}-{str(index).zfill(5)}-{label}-{zaxis}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = np.frombuffer(buffer.read(), dtype=np.float32).reshape((self.resolution, self.resolution, -1))#Image.open(buffer)
        #img = np.repeat(img[:,:,0:1], 3, axis=2)
        img = self.transform(img)

        return img, label, int(zaxis)# // 10
