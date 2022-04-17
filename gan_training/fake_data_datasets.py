from torch.utils.data import DataLoader,Dataset
import os
import numpy as np
from PIL import Image
from torchvision import transforms
import torch

class FakeData(Dataset):
    def __init__(self, img_path, label_path, transform=None):
        self.img_path = img_path
        self.latentvecs_dict = np.load(label_path, allow_pickle=True).item()
        self.transform = transform
        self.img_name_list = os.listdir(img_path)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        img_name = self.img_name_list[idx]
        image = Image.open(os.path.join(self.img_path, img_name))
        image = self.transform(image)
        latentvec = self.latentvecs_dict[img_name]
        return image, torch.FloatTensor(latentvec)



if __name__ == '__main__':
    size = 128
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    dataset=FakeData(img_path='data/imagenet_fake_data/image', label_path='data/imagenet_fake_data/latentvecs.npy', transform=transform)
    train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    for step ,(image, latentvec) in enumerate(train_loader):
        print(image)
        print(latentvec.shape)
        break