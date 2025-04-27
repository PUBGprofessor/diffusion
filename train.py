from dataset import get_dataloader
from model.DDPM import DDPM
from model.Net import build_network, unet_res_cfg

import os
import torch
import torch.nn as nn
batch_size = 64
n_epochs = 100


def train(ddpm: DDPM, net, device, ckpt_path):
    # n_steps 就是公式里的 T
    # net 是某个继承自 torch.nn.Module 的神经网络
    n_steps = ddpm.n_steps
    dataloader = get_dataloader(batch_size)
    net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    for e in range(n_epochs):
        for x, _ in dataloader:
            current_batch_size = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, n_steps, (current_batch_size, )).to(device)
            eps = torch.randn_like(x).to(device)
            x_t = ddpm.sample_forward(x, t, eps)
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))
            loss = loss_fn(eps_theta, eps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(net.state_dict(), ckpt_path)

if __name__ == '__main__':
    n_steps = 100
    config_id = 4
    device = 'cuda'
    model_path = 'output/model_unet_res.pth'
    os.makedirs('output', exist_ok=True)

    config = unet_res_cfg
    net = build_network(config, n_steps)
    ddpm = DDPM(device, n_steps)

    train(ddpm, net, device=device, ckpt_path=model_path)

