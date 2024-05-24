from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np

def one_hot(seg, num_classes=2):
    return np.eye(num_classes)[np.rint(seg).astype(int)].astype(np.float32)


class InferenceDataset(Dataset):

    def __init__(self, root, opts, transform=None):
        self.paths = sorted(data_utils.make_dataset(root))
        self.transform = transform
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        from_im = np.load(from_path) #from_im = Image.open(from_path)
        
        img, msk = from_im[:,:,:1], from_im[:,:,1:]
        msk = np.argmax(msk, axis=-1)
        msk = 0*msk#msk = np.where(msk == 2, 1, msk)
        msk = one_hot(msk, num_classes=3)
        msk = np.where(np.isclose(msk, 1), 1., -1.)
        img = np.concatenate([img, msk], axis=-1)
        
        if self.transform:
            img = self.transform(img)

        labels = np.array([1,0]) if "acq-TOF" in from_path else np.array([0,1])

        return img, labels