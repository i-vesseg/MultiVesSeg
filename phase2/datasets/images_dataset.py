import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
from utils import data_utils
import numpy as np
import random
import os

def one_hot(seg, num_classes=2):
    return np.eye(num_classes)[np.rint(seg).astype(int)].astype(np.float32)

class ImagesDataset(Dataset):

    def __init__(self, opts, source_root, target_roots=None, transform=None, one_target_slice=False):
        if source_root is not None:
            self.paths_tof = sorted(data_utils.make_dataset(source_root))
        else:
            self.paths_tof = []
        
        if target_roots is not None:
            self.paths_labeled_swi = sorted(data_utils.make_dataset(target_roots["labeled"]))
            self.paths_swi = self.paths_labeled_swi[:]
            self.paths_labeled_swi = self.discard_zero_weights(self.paths_labeled_swi)
            if one_target_slice:
                self.paths_labeled_swi = self.one_slice_per_volume(self.paths_labeled_swi)
            if "unlabeled" in target_roots:
                self.paths_swi += sorted(data_utils.make_dataset(target_roots["unlabeled"]))
        else:
            self.paths_swi = []
        
        self.len_tof = len(self.paths_tof)
        self.len_swi = len(self.paths_swi)
        
        if self.len_swi == 0 and self.len_tof == 0:
            raise Exception("Dataset is empty")
        elif self.len_swi == 0: #only tof
            self.len_total = self.len_tof
        elif self.len_tof == 0: #only swi
            self.len_total = self.len_swi
        else:
            self.len_total = max(self.len_tof, self.len_swi) * 2
            if self.len_tof > self.len_swi:
                self.len_total -= 1
        
        self.transform = transform
        self.opts = opts

    def __len__(self):
        if "batch_size" in self.opts and "test_batch_size" in self.opts:
            return max(self.len_total, self.opts.batch_size, self.opts.test_batch_size)
        return self.len_total

    def __getitem__(self, index):
        if len(self) != self.len_total:
            index = index % self.len_total
        
        if self.len_swi == 0: #only tof
            img_path = self.paths_tof[index]
            domain_label = 0
        elif self.len_tof == 0: #only swi
            img_path = self.paths_swi[index]
            domain_label = 1
        else:
            if index % 2 == 0:
                img_path = self.paths_tof[index // 2 % self.len_tof]
                domain_label = 0
            else:
                img_path = self.paths_swi[index // 2 % self.len_swi]
                domain_label = 1

        sample = np.load(img_path, allow_pickle=True).item()
        img, msk = sample["data"], sample["mask"]
        if len(img.shape) == 2:
            img = img[:,:,None]
        
        has_gt = True
        if domain_label == 1 and img_path not in self.paths_labeled_swi:
            has_gt = False
            msk = np.logical_and(msk, False)

        if self.opts.invert_intensity:
            if domain_label == 1:
                img *= -1
            else:
                img = np.where(np.argmax(msk, axis=-1, keepdims=True) == 0, -1*img, img)
        
        if "weight" not in sample or sample["weight"] is None:
            weight = np.ones(img.shape, dtype=msk.dtype)
        else:
            weight = sample["weight"]
            
        if self.transform:
            sample = self.transform({"data": img, "mask": msk, "weight": weight})
        
        return sample["data"], sample["mask"], sample["weight"], one_hot(domain_label, num_classes=2), has_gt
    
    def discard_zero_weights(self, labeled_paths):
        non_zero = []
        for labeled_path in labeled_paths:
            sample = np.load(labeled_path, allow_pickle=True).item()
            if "weight" in sample and sample["weight"] is not None and np.all(sample["weight"] == 0):
                continue
            if np.all(np.argmax(sample["mask"], axis=-1) == 0):
                continue
            non_zero.append(labeled_path)
        return non_zero
    
    def one_slice_per_volume(self, labeled_paths):
        slices = {}
        for labeled_path in labeled_paths:
            patient_id = os.path.basename(labeled_path).split("_slice")[0]
            if patient_id not in slices:
                slices[patient_id] = []
            slices[patient_id].append(labeled_path)
        return sum([[s[len(s)//2]] for s in slices.values()], start=[])#1slice per volume
        #return sum([[s[len(s)//3]] + [s[len(s)//2]] + [s[len(s)*2//3]] for s in slices.values()], start=[])#3slices per volume

class MyRandomSampler(Sampler):
    def __init__(self, data_source, balanced_sampling=True):
        self.data_source = data_source
        self.lucky_paths = np.array([i for i, p in enumerate(data_source.paths_swi) if p in data_source.paths_labeled_swi])
        self._num_samples = len(self.data_source)
        self.replacement = False
        self.generator = None
        self.balanced_sampling = balanced_sampling

    @property
    def num_samples(self):
        return self._num_samples
    
    def __len__(self):
        return self.num_samples
    
    def mix_iter(self):
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        idx_list = [
            idx for pair in torch.randperm(self.num_samples // 2, generator=generator).tolist() for idx in (pair*2, pair*2+1)
        ]
        
        if len(self.lucky_paths) > 0 and self.balanced_sampling:
            new_idx_list = idx_list[:]
            bucket = []
            for i in range(len(idx_list)//4):
                batch = idx_list[i*4 : (i+1)*4]
                semi_ids = [id for id in batch if id in (1 + self.lucky_paths*2)]
                if len(semi_ids) == 0:
                    bucket += [new_idx_list[i*4+1]]
                    new_idx_list[i*4+1] = 1 + random.sample(self.lucky_paths.tolist(), 1)[0]*2
                elif len(semi_ids) == 2 and len(bucket) > 0:
                    new_idx_list[i*4+3] = bucket.pop(0)
            idx_list = new_idx_list
        
        if len(idx_list) < self.num_samples:
            idx_list += [self.num_samples - 1]

        return idx_list

    def __iter__(self):
        if self.data_source.len_tof > 0 and self.data_source.len_swi > 0:
            yield from self.mix_iter()
        else:
            yield from random.sample(list(range(self.num_samples)), self.num_samples)