from dataset import get_dataloader
from model.DDPM import DDPM
from model.UNet import MyUNet_w_pe, UNet
# from diffusion.model.ConvNet import build_network, unet_res_cfg
from utils.image import get_img_shape, set_img_size, show_images

import einops
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import time
# from torch.nn.parallel import DataParallel

batch_size = 1024
n_epochs = 1000

set_img_size((1, 28, 28))

def train(ddpm: DDPM, net, device, ckpt_path):
    # n_steps 就是公式里的 T
    net.train()
    # net 是某个继承自 torch.nn.Module 的神经网络
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    start_time = time.time()

    for e in range(n_epochs):
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
        print(f"Epoch: {e}, Loss: {loss_sum / len(dataloader)}, Time: {epoch_end_time - epoch_start_time:.2f}s")
        if (e + 1) % 50 == 0:
            # save model
            torch.save(net.state_dict(), os.path.join(ckpt_path, f"epoch_{e + 1}.pth"))
        
    end_time = time.time()
    print(f"Sum Time: {end_time - start_time:.2f}s")
    print("Done training!")

if __name__ == '__main__':
    n_steps = 1000
    config_id = 4
    device = 'cuda'
    model_path = './output'
    os.makedirs('output', exist_ok=True)

    # config = unet_res_cfg
    # net = build_network(config, n_steps)
    net = MyUNet_w_pe(n_steps)
    # net = DataParallel(net, device_ids=[0, 1]) 
    # net = UNet(n_steps,[32, 64, 128 ,256], pe_dim=20, residual=True)
    # net = MyUNet_w_pe(n_steps, ch_list=[4, 8, 16, 32])
    ddpm = DDPM(device, n_steps)
    # torch.save(net.state_dict(), os.path.join(model_path, f"MyUNet_w_pe2.pth"))
    train(ddpm, net, device=device, ckpt_path=model_path)
    # for i, _ in get_dataloader(64):
    #     print(i.shape)
    #     show_images(i, './output/sampe.png')
    #     break


