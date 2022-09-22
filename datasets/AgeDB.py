import os
import logging
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from torch.utils import data
import torchvision.transforms as transforms


class AgeDB(data.Dataset):
    def __init__(self, df, data_dir, img_size, split='train', reweight='none',
                 lds=False, lds_kernel='gaussian', lds_ks=5, lds_sigma=2):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split


    def __len__(self):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['age']]).astype('float32')
        weight = np.asarray([self.weights[index]]).astype('float32') if self.weights is not None else np.asarray([np.float32(1.)])

        return img, label, weight
