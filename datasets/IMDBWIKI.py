from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import math
import os


class IMDBWIKI(data.Dataset):
    def __init__(self, df, data_dir, img_size = 224, split='train', group_num = 10):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split    
        self.group_range = len(self.df)/group_num
        self.key_list = [i for i in range(group_num)]
        # key is the group is, value is the group num
        self.group_dict = {}
        if split == 'train':
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                age = row['age']
                group_id = math.ceil(age/self.group_range)
                if group_id in self.group_dict.keys():
                    self.group_dict[group_id] += 1
                else:
                    self.group_dict[group_id] = 1
        else:
            pass
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['age']]).astype('float32')
        group_index = math.ceil(label/self.group_range) 
        group = np.asarray([group_index]).astype('float32')
        return img, label, group

    def get_group(self):
        return self.group_dict

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform