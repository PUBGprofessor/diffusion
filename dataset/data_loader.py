import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Lambda
from dataset.mnist import load_mnist

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)
    # print(t_train.shape, t_test.shape)
    return x_train, t_train, x_test, t_test

class Dataset():
    def __init__(self, device, transform=None):
        self.transform = transform
        self.x_train = torch.tensor(get_data()[0], device=device)
        self.x_train = self.transform(self.x_train)
        self.device = device
    
    def __len__(self):
        return self.x_train.shape[0]
    
    def __getitem__(self, index):
        # if index < self.x_train.shape[0]:
        #     return self.x_train[index], torch.tensor(1.0, device=self.device)
        # else:
        #     return self.G.getImage("test"), torch.tensor(0.0, device=self.device)
        return self.x_train[index], torch.tensor(1.0, device=self.device)

def get_dataset(device: str = 'cuda'):
    transform = Compose([Lambda(lambda x: (x - 0.5) * 2)])
    # transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = Dataset(device = device, transform=transform)
    return dataset

def get_dataloader(batch_size: int, device: str = 'cuda'):
    transform = Compose([Lambda(lambda x: (x - 0.5) * 2)])
    # transform = Compose([ToTensor(), Lambda(lambda x: (x - 0.5) * 2)])
    dataset = Dataset(device='cuda', transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
