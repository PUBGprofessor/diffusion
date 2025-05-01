from dataset import get_dataset, get_dataloader
from model.DDPM import DDPM
from model.UNet import MyUNet_w_pe, UNet
# from diffusion.model.ConvNet import build_network, unet_res_cfg
from utils.image import get_img_shape, set_img_size, show_images, sample_imgs

import einops
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

batch_size = 512
n_epochs = 3000
n_steps = 1000
world_size = 2
# device = 'cuda'
model_path = './output/v3'
load_epoch = 0

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

def train_fn(rank, world_size):
    set_img_size((3, 64, 64))

    print(f"Running basic DDP example on rank {rank}.")
    device = torch.device(f"cuda:{rank}")
    # initialize the process group
    dist.init_process_group("NCCL", rank=rank, world_size=world_size)

    net = MyUNet_w_pe(n_steps).to(device)

    net = DDP(net, device_ids=[rank])
    if load_epoch != 0:
        net.load_state_dict(torch.load(model_path + f"/epoch_{load_epoch}.pth"))
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device=device, ckpt_path=model_path, rank=rank)

    dist.destroy_process_group()


def train(ddpm: DDPM, net, device, ckpt_path, rank):
    # n_steps 就是公式里的 T
    # net.train()
    n_steps = ddpm.n_steps
    # dataloader = get_dataloader(batch_size, device)
    dataset = get_dataset(device=device)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True, prefetch_factor=8)
    # net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-4)

    start_time = time.time()

    for e in range(load_epoch, n_epochs):
        net.train()
        epoch_start_time = time.time()
        loss_sum = 0
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            loss = loss_fn(eps_theta, eps)
            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_end_time = time.time()
        if rank == 0:
            print(f"Epoch: {e}, Loss: {loss_sum / len(dataloader)}, Time: {epoch_end_time - epoch_start_time:.2f}s")
            ep = e + 1
            if ep % 100 == 0:
                # save model
                with torch.no_grad():
                    torch.save(net.state_dict(), os.path.join(ckpt_path, f"epoch_{e + 1}.pth"))
                    net.eval()
                    sample_imgs(ddpm, net, os.path.join(ckpt_path, f"epoch_{e + 1}.png"), n_sample=81, device=device)

    if rank == 0: 
        end_time = time.time()
        print(f"Sum Time: {end_time - start_time:.2f}s")
        print("Done training!")


if __name__ == '__main__':
    # set_img_size((1, 28, 28))
    os.environ["MASTER_ADDR"] = "localhost"# ——11——
    os.environ["MASTER_PORT"] = "29500"
    os.makedirs('output', exist_ok=True)

    mp.spawn(train_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

