from dataset import get_dataloader
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
# from torch.nn.parallel import DataParallel

batch_size = 256
n_epochs = 1000
n_steps = 500
ch_list=[16, 32, 64, 128]
pe_dim = 32
# ch_list=[32, 64, 128 ,256]
world_size = 2
# device = 'cuda'
model_path = './output/v4'
save_img_path = "output/v4/test"
load_epoch = 0

set_img_size((3, 64, 64))


def test(ddpm, net, device, ckpt_path):
    with torch.no_grad():
        net.eval()
        sample_imgs(ddpm, net, os.path.join(ckpt_path, f"test.png"), n_sample=81, device=device)

if __name__ == '__main__':
    set_img_size((3, 64, 64))

    os.makedirs(save_img_path, exist_ok=True)
    device = "cuda"
    # config = unet_res_cfg
    # net = build_network(config, n_steps)
    net = MyUNet_w_pe(n_steps, ch_list, pe_dim).to(device)
    # net = DataParallel(net, device_ids=[0, 1]) 
    # net = UNet(n_steps,[32, 64, 128 ,256], pe_dim=20, residual=True)
    # net = MyUNet_w_pe(n_steps, ch_list=[4, 8, 16, 32])
    ddpm = DDPM(device, n_steps)
    state_dict = torch.load(os.path.join(model_path, f"epoch_{n_epochs}.pth"))
    # 去掉 'module.' 前缀（适用于从 DDP 保存的模型）
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net.load_state_dict(new_state_dict)
    # torch.save(net.state_dict(), os.path.join(model_path, f"MyUNet_w_pe2.pth"))
    test(ddpm, net, device=device, ckpt_path=save_img_path)
    # for i, _ in get_dataloader(64):
    #     print(i.shape)
    #     show_images(i, './output/sampe.png')
    #     break


