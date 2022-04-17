import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from gan_training.fake_data_datasets import FakeData

def get_dataset(name, data_dir, label_path=None, size=64, lsun_categories=None, simple_transform=False, pretrained_transform=False):
    if simple_transform:
        transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif pretrained_transform:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x + 1./128 * torch.rand(x.size())),
        ])

    if name == 'image':
        dataset = datasets.ImageFolder(data_dir, transform)
        nlabels = len(dataset.classes)
    elif name == 'npy':
        # Only support normalization for now
        dataset = datasets.DatasetFolder(data_dir, npy_loader, ['npy'])
        nlabels = len(dataset.classes)
    elif name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                   transform=transform)
        nlabels = 10
    elif name == 'lsun':
        if lsun_categories is None:
            lsun_categories = 'train'
        dataset = datasets.LSUN(data_dir, lsun_categories, transform)
        nlabels = len(dataset.classes)
    elif name == 'lsun_class':
        dataset = datasets.LSUNClass(data_dir, transform,
                                     target_transform=(lambda t: 0))
        nlabels = 1
    elif name == 'imagenet fake data':
        dataset = FakeData(img_path=data_dir, label_path=label_path, transform=transform)
        nlabels = 1000
    else:
        raise NotImplemented

    return dataset, nlabels


def npy_loader(path):
    img = np.load(path)

    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        img = img/127.5 - 1.
    elif img.dtype == np.float32:
        img = img * 2 - 1.
    else:
        raise NotImplementedError

    img = torch.Tensor(img)
    if len(img.size()) == 4:
        img.squeeze_(0)

    return img
