import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from data import get_train_transform, get_test_transform


class CustomDataset(Dataset):
    img_aug = True
    imgs = []
    transform = None

    def __init__(self, label_file, image_set, input_size):
        with open(label_file, 'r') as f:
            self.imgs = list(map(lambda line: line.strip().split('|'), f))

        if image_set == 'train':
            self.transform = get_train_transform(size=input_size)
        else:
            self.transform = get_test_transform(size=input_size)
        self.input_size = input_size

    def __getitem__(self, index):
        # print(self.imgs)
        # print(index)
        # print(len(self.imgs[index]))
        img_path, label = self.imgs[index]
        # print(img_path)
        img = Image.open(img_path).convert('RGB')
        if self.img_aug:
            img = self.transform(img)
        else:
            img = np.array(img)
            img = torch.from_numpy(img)

        return img, torch.from_numpy(np.array(int(label)))
 
    def __len__(self):
        return len(self.imgs)


def get_datasets_and_dataloader(label_path, image_set, batch_size, input_size):
    _dataset = CustomDataset(label_path, image_set=image_set, input_size=input_size)
    _dataloader = DataLoader(_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return _dataset, _dataloader
