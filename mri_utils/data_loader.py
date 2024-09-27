import cv2
import torch
import numpy as np
from glob import glob
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

__DATASET__ = {}

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

class Padding:
    def __init__(self, tgt_size):
        self.tgt_size = tgt_size
        assert(len(tgt_size) == 2 and tgt_size[0] == tgt_size[1])

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        if img.shape[0] == self.tgt_size[0] and img.shape[1] == self.tgt_size[1]:
            return img 
        
        y_pad = self.tgt_size[0]-img.shape[0]
        x_pad = self.tgt_size[1]-img.shape[1]
        img_rt = np.pad(img, ((y_pad//2, y_pad//2 + y_pad%2), (x_pad//2, x_pad//2 + x_pad%2)), mode = 'constant')
        assert(img_rt.shape[0] == self.tgt_size[0] and img_rt.shape[1] == self.tgt_size[1])

        return img_rt
    


class Crop:
    'crop an image to target size'
    def __init__(self, tgt_size):
        self.tgt_size = tgt_size
        assert(len(tgt_size) == 2 and tgt_size[0] == tgt_size[1])
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        assert(img.shape[0] >= self.tgt_size[0] and img.shape[1] >= self.tgt_size[1])
        y_start = (img.shape[0] - self.tgt_size[0])//2
        x_start = (img.shape[1] - self.tgt_size[1])//2
        img_rt = img[y_start:y_start+self.tgt_size[0], x_start:x_start+self.tgt_size[1]]
        assert(img_rt.shape[0] == self.tgt_size[0] and img_rt.shape[1] == self.tgt_size[1])
        return img_rt
    
class Resize:
    'resize an image to target size'
    def __init__(self, tgt_size):
        self.tgt_size = tgt_size
        assert(len(tgt_size) == 2 and tgt_size[0] == tgt_size[1])
    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        img_rt = cv2.resize(img, (self.tgt_size[0], self.tgt_size[1]))
        assert(img_rt.shape[0] == self.tgt_size[0] and img_rt.shape[1] == self.tgt_size[1])
        return img_rt

class NpToTensor:
    def __init__(self):
        pass
    def __call__(self, img):
        if img.dtype == 'uint16':
            img = (img/65535.0).astype(np.float32) # as we store image as uint16, the reformatted float32 shouldn't loss information
        elif img.dtype == 'uint8':
            img = (img/255.0).astype(np.float32)
        else:
            raise NotImplementedError
        # img = (2*img)-1  # 10192023, normalize to [-1, 1]

        return torch.Tensor(img)

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader

@register_dataset(name='mri')
class MRIDataset(VisionDataset):
    def __init__(self, root: str, im_sz: int):
       
        super().__init__(root)
        self.im_sz = im_sz
        self.transforms = Compose([Resize((im_sz, im_sz)), 
                                    NpToTensor()])

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
         
        
        if self.transforms is not None:
            img = self.transforms(img)
        img = img.reshape((1, self.im_sz, self.im_sz))
        # repeat the first channel
        # img = torch.cat((img, img, img), dim=0)
        # print('input min: {}, max: {}'.format(img.min(), img.max()))
        
        return img
    
def adjust_gamma(image, gamma=2.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


@register_dataset(name='mri_test')
class MRITestDataset(VisionDataset):
    def __init__(self, root: str, im_sz: int):
       
        super().__init__(root)
        self.im_sz = im_sz
        self.transforms = Compose([Resize((im_sz, im_sz)), 
                                    NpToTensor()])

        self.fpaths = sorted(glob(root + '/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
         
        
        if self.transforms is not None:
            img = self.transforms(img)
        img = img.reshape((1, self.im_sz, self.im_sz))
       
        return img
    
def adjust_gamma(image, gamma=2.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


