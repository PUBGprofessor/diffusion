"""
    fast image loader from https://www.kaggle.com/code/money0/anime-diffusion/notebook
"""
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
# from torchvision.transforms import Compose, ToTensor, Lambda
import torchvision.transforms as tt

DATA_DIR = './anime-faces'

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# class Dataset():
#     def __init__(self, dataset, device):
#         self.dataset = dataset
#         self.device = device
    
#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, index):
#         return self.dataset[index][0], torch.tensor(1.0, device=self.device)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def get_dataset(device:str="cuda"):
    image_size = 64
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    train_ds = ImageFolder(DATA_DIR, transform=tt.Compose([
        tt.Resize(image_size),
        tt.CenterCrop(image_size),
        tt.ToTensor(), # 将 PIL 图像转为 FloatTensor，范围 [0,1]
        tt.Normalize(*stats)])) # 等比缩放到[-1, 1]
    
    return train_ds
    # return Dataset(train_ds, device)

def get_dataloader(batch_size, device:str="cuda", shuffle=False):
    train_ds = get_dataset()
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return DeviceDataLoader(train_dl, device)

# print(get_Dataset("cuda")[0])
