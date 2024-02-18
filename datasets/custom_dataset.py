from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
import random


class CubDataset(Dataset):
    def __init__(self, txt_file, root_dir, enhance_transform, co_transform, crop_transform, training=False):
        self.image_list = []
        self.id_list = []
        self.root_dir = root_dir
        self.enhance_transform = enhance_transform
        self.co_transform = co_transform
        self.crop_transform = crop_transform
        self.num_classes = 0
        self.training = training
        with open(txt_file, 'r') as f:
            line = f.readline()
            # self.datas = f.readlines()
            while line:
                img_name = line.split()[0]
                label = int(line.split()[1])
                # label = int(label)
                self.image_list.append(img_name)
                self.id_list.append(label)
                line = f.readline()
        self.num_classes = max(self.id_list) + 1

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = self.id_list[idx]
        img_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_name).convert('RGB')

        if self.enhance_transform:
            image = self.enhance_transform(image)
        crop = self.crop_transform(image)
        if self.co_transform:
            image = self.co_transform(image)
            crop = self.co_transform(crop)
        return image, crop, label


def test_dataset():
    root = '/home/zengh/Dataset/Fine-grained/CUB_200_2011/images'
    txt = '/home/zengh/Dataset/Fine-grained/CUB_200_2011/test_pytorch.txt'
    from torchvision import transforms
    rgb_mean = [0.5, 0.5, 0.5]
    rgb_std = [0.5, 0.5, 0.5]
    size = 448
    enhance_transform = transforms.Compose([
        transforms.Resize((size, size)),
    ])
    co_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize(rgb_mean, rgb_std), ])
    corp_transform = transforms.Compose([transforms.FiveCrop(size * 0.7)])
    carData = CubDataset(txt, root, enhance_transform, co_transform, corp_transform, True)
    dataloader = DataLoader(carData, batch_size=16, shuffle=True)
    for data in dataloader:
        images, labels = data
        # print(images.size(),labels.size(),labels)


if __name__ == '__main__':
    test_dataset()
