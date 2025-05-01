import torch
import os
import glob
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from PIL import Image
import numpy as np

path = r"anime-faces/data"
save_path = r"anime-faces/pkldata"

def get_data():
    files = glob.glob(path + "/*png")
    data = []
    for png_path in files:
        img = Image.open(png_path).convert("RGB")
        data.append(np.array(img))
    
    return data

class Dataset():
    def __init__(self, device, transform):
        self.data = torch.tensor(get_data(), dtype=torch.float32, device=device)
        self.transform = transform
        self.data = self.transform(self.data)
        self.device = device
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], torch.tensor(1.0, device=self.device)

def get_dataset(device:str="cuda"):
    transform = Compose([lambda x : (x - 128) / 128])  # 归一化到[-1, 1]
    return Dataset(device, transform)

def get_dataloader(batch_size, device:str="cuda", shuffle=False):
    transform = Compose([lambda x : (x - 128) / 128])  # 归一化到[-1, 1]
    dataset = Dataset(device, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# print(get_Dataset()[0])