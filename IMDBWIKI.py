from torch.utils import data
import torchvision.transforms as transforms


class IMDBWIKI(data.Dataset):
    def __init__(self, df, data_dir, img_size = 224, split='train'):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row['path'])).convert('RGB')
        transform = self.get_transform()
        img = transform(img)
        label = np.asarray([row['age']]).astype('float32')

        return img, label

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