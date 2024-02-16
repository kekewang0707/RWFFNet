import os

from torchvision import datasets
from torchvision.transforms import transforms

CIFAR10_PATH = 'CIFAR-10'
IMAGENET_PATH = 'imagenet'
IMAGENET100_PATH = 'imagenet100'
CIFAR100_PATH = 'CIFAR-100'
FREPlus_PATH = 'data'
PASCALVOC07_PATH = 'pascal voc2007'


def create_dataset(dataset_name, subset):
    assert dataset_name in ['cifar10', 'imagenet', 'cifar100', 'imagenet100', 'freplus', 'pascal voc2007']
    assert subset in ['train', 'val']

    if dataset_name == 'freplus':
        if subset == 'train':
            return datasets.ImageFolder(os.path.join(FREPlus_PATH, 'Training'),
                                        transform=transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),
                                        ]))
        else:
            return datasets.ImageFolder(os.path.join(FREPlus_PATH, 'PublicTest'),
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),
                                        ]))
    elif dataset_name == 'pascal07':
        if subset == 'train':
            return datasets.VOCDetection(PASCALVOC07_PATH, year='2007', image_set="train",
                                         transform=transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.RandomCrop(224),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ]))
        else:
            return datasets.VOCDetection(PASCALVOC07_PATH, year='2007', image_set="val",
                                         transform=transforms.Compose([
                                             transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                         ]))
    if dataset_name == 'imagenet100':
        if subset == 'train':
            return datasets.ImageFolder(os.path.join(IMAGENET100_PATH, 'train'),
                                        transform=transforms.Compose([
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),
                                        ]))
        else:
            return datasets.ImageFolder(os.path.join(IMAGENET100_PATH, 'val'),
                                        transform=transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225]),
                                        ]))
    elif dataset_name == 'cifar10':
        if subset == 'train':
            return datasets.CIFAR10(CIFAR10_PATH, train=True, download=False,
                                    transform=transforms.Compose([
                                        transforms.Pad(4),
                                        transforms.RandomCrop(32),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))]))
        else:
            return datasets.CIFAR10(CIFAR10_PATH, train=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))]))
    elif dataset_name == 'cifar100':
        if subset == 'train':
            return datasets.CIFAR100(CIFAR100_PATH, train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.Pad(4),
                                         transforms.RandomCrop(32),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]))
        else:
            return datasets.CIFAR100(CIFAR100_PATH, train=False, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ]))
    else:
        raise ValueError('??')
