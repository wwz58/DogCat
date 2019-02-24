import random
import os
from torch.utils import data
from PIL import Image


class DogCat(data.Dataset):
    def __init__(self, root, train=True, test=False, train_ration=0.7, transform=None):
        super(DogCat, self).__init__()
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        if not test:
            random.shuffle(imgs)
            train_num = int(len(imgs) * train_ration)
            if train:
                imgs = imgs[:train_num]
            else:
                imgs = imgs[train_num:]
        self.num = len(imgs)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        if not self.test:
            label = 1 if 'dog' in img_path.split('\\')[-1] else 0
        else:
            label = img_path.split('\\')[-1].split('.')[0]
        return img, label

    def __len__(self):
        return self.num