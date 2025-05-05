# 动手实践 Diffusion 模型：从手写数字到动漫人脸图像生成

> 本文记录了我从零实现扩散模型（Diffusion Model），并应用于 MNIST 手写数字和 Anime Face 动漫人脸生成的完整过程。内容涵盖基础理论、模型实现、训练过程、效果可视化和踩坑总结。

------

## 一、什么是 Diffusion 模型？

扩散模型（Diffusion Model）是一类通过逐步添加噪声扰乱图像，并学习逆过程还原的生成模型。近年来在图像生成领域表现突出，代表作品包括 DDPM、Stable Diffusion 等。

原始论文：https://arxiv.org/abs/2006.11239

![DDPM](\source\DDPM.png)

其核心思想：

1. **前向过程（Forward Process）**：逐步将真实图像加噪，变成纯噪声。

   在前向过程中，来自训练集的图像x0会被添加T次噪声，使得xT为符合标准正态分布。准确来说，「加噪声」并不是给上一时刻的图像加上噪声值，而是从一个均值与上一时刻图像相关的正态分布里采样出一幅新图像。

   ![image-20250502112723413](\source\image-20250502112723413.png)

   令αt=1−βt,α¯t=∏i=1tαi，则：

   ![image-20250502112756171](\source\image-20250502112756171.png)

   当step足够多、β逐渐增大时，最后的x_t则接近正态分布

2. **反向过程（Reverse Process）**：通过一个神经网络（通常是 UNet），一步步预测并去除噪声，从噪声还原图像。

   这里的推导比较复杂，我也没有完全看懂，最后的均值分布是：

   ![image-20250502113308018](\source\image-20250502113308018.png)

   那就简单来说：噪声当然是不可逆的，因此我们用一个神经网络来预测：

   **网络输入（有噪声的图像，当前是哪一步）， 然后预测出（下一步去噪后的图像**）

   可能这样对神经网络来说还是太难了，因此我们直接让他预测在这一步的前向过程中加的标准正态分布的噪声是什么，然后我们自己算图像x_t-1:

   **网络输入（有噪声的图像，当前是哪一步）， 然后预测出（这一步的噪声**）

## 二、DDPM实现代码

DDPM类即上面所说的前向反向过程的主体控制逻辑类，可以说50行包含了整个扩散模型一篇论文的内容，其他的模块（如数据集、网络）都可以替换

代码参考https://zhouyifan.net/2023/07/07/20230330-diffusion-model/



```python
import torch

class DDPM():   
    # n_steps 就是论文里的 T
    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device) # 线性增加的beta [0.0001, 0.02]
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
	
	# 正向过程可以根据上面的公式一步计算出来任意步的x_t
    def sample_forward(self, x, t, eps=None):
        # 加噪
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) # 扩展到和x一样的形状[batch_size, 1, 28, 28] 
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res
    
    # 反向过程需要一步步用选取的网络预测
    def sample_backward(self, img_shape, net, device, simple_var=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                    1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        mean = (x_t -
                (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                eps) / torch.sqrt(self.alphas[t])
        # mean = (x_t -
        #         # (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
        #         eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t
```

## 三、时间步编码

因为网络对每一步的图形去噪的方式是不一样的，因此需要让网络知道现在是哪一步。

使用同transformer的时间编码，不过Embedding是固定的：

```
class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int=1000, d_model: int=20):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)  # [0., 1., 2., ..., max_seq_len - 1.]
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)  # [0., 2., 4., ..., d_model - 2]
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)
```

然后在每一层的的网络中，先用一个Linear网络将维度为d_model(上面为20)投影为当前的channels数量，然后在每个channel上分别加上这个时间编码（同transformer在序列中的每个单词的每个维度上加上位置编码），具体见下面的UNet

## 四、UNet实现

![v2-68ffbaff593f95cc96fc4b6811356e39_r](\source\v2-68ffbaff593f95cc96fc4b6811356e39_r.jpg)

```python
class UnetBlock(nn.Module):
    def __init__(self, shape, in_ch, out_ch, residual=True):
        super().__init__()

        self.LayerNorm = nn.LayerNorm(shape)
        self.activation = nn.LeakyReLU()
        self.gen = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.residual = residual
        if residual:
            if in_ch == out_ch:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_ch, out_ch, 1)

class MyUNet_w_pe(nn.Module):
    def __init__(self, n_steps, ch_list=[32, 64, 128 ,256], pe_dim=20, residual=True):
        super().__init__()
        image_size = get_img_shape()
        self.pe =  PositionalEncoding(n_steps, pe_dim)
        C, H, W = image_size
        self.H = [H,]
        self.W = [W,]
        for i in range(len(ch_list) + 1): # 0, 1, 2, 3, 4
            self.H.append(self.H[-1] // 2)
            self.W.append(self.W[-1] // 2)
        # print(self.W)
        
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()

        # encoder最上面第一层UNetBlock
        self.encoder.append(nn.Sequential(
            UnetBlock((C, H, W), C, ch_list[0], True),
            UnetBlock((ch_list[0], H, W), ch_list[0], ch_list[0], True),
            # F.max_pool2d(),
        ))
        self.pe_linears_en.append(nn.Sequential(
            nn.Linear(pe_dim, C),
            nn.ReLU(),
            nn.Linear(C, C)
        ))

        # decoder最上面第一层UNetBlock
        self.decoder.append(nn.Sequential(
            UnetBlock((2 * ch_list[0], H, W), 2 * ch_list[0], ch_list[0], True),
            UnetBlock((ch_list[0], H, W), ch_list[0], ch_list[0], True),
            nn.Conv2d(ch_list[0], C, 1)
        ))
        self.pe_linears_de.append(nn.Sequential(
            nn.Linear(pe_dim, 2 * ch_list[0]), 
            nn.ReLU(),
            nn.Linear(2 * ch_list[0], 2 * ch_list[0])
        ))
        """
        反卷积计算公式:(H_in - 1 * S - 2P + D * (K - 1) + O + 1
        因此归定:偶数时padding为1,outpading为1
                奇数时padding为0,outpading为0
                确保一致
        """
        for i in range(1, len(ch_list)):
            # 此时输入为[b, ch_list[i - 1], self.H[i - 1], self.W[i - 1]]
            # 输出为[b, ch_list[i], self.H[i], self.W[i]]
            self.encoder.append(nn.Sequential(
                nn.MaxPool2d(2),
                UnetBlock((ch_list[i - 1], self.H[i], self.W[i]), ch_list[i - 1], ch_list[i], True),
                UnetBlock((ch_list[i], self.H[i], self.W[i]), ch_list[i], ch_list[i], True),
                # F.max_pool2d(),
            ))
            self.pe_linears_en.append(nn.Sequential(
                nn.Linear(pe_dim, ch_list[i - 1]),
                nn.ReLU(),
                nn.Linear(ch_list[i - 1], ch_list[i - 1])
            ))

            # 此时输入为[b, 2 * ch_list[i], self.H[i], self.W[i]]
            # 输出为[b, ch_list[i-1], self.H[i - 1], self.W[i - 1]]
            H_padding = 1 if (self.H[i - 1] % 2 == 0) else 0
            W_padding = 1 if (self.W[i - 1] % 2 == 0) else 0
            self.decoder.append(nn.Sequential(
                UnetBlock((2 * ch_list[i], self.H[i], self.W[i]), 2 * ch_list[i], ch_list[i], True),
                UnetBlock((ch_list[i], self.H[i], self.W[i]), ch_list[i], ch_list[i], True),
                nn.ConvTranspose2d(ch_list[i], ch_list[i - 1], 3, 2, (H_padding, W_padding), (H_padding, W_padding))
            ))
            self.pe_linears_de.append(nn.Sequential(
                nn.Linear(pe_dim, 2 * ch_list[i]), 
                nn.ReLU(),
                nn.Linear(2 * ch_list[i], 2 * ch_list[i])
            ))

        # 输入 [b, ch_list[-1], self.H[-2], self.W[-2]]
        # 输出 [b, ch_list[-1], self.H[-2], self.W[-2]]
        H_padding = 1 if (self.H[-2] % 2 == 0) else 0
        W_padding = 1 if (self.W[-2] % 2 == 0) else 0
        self.bottom = nn.Sequential(
            nn.MaxPool2d(2),
            UnetBlock((ch_list[-1], self.H[-2], self.W[-2]), ch_list[-1], 2 * ch_list[-1], True),
            UnetBlock((2 * ch_list[-1], self.H[-2], self.W[-2]), 2 * ch_list[-1], 2 * ch_list[-1], True),
            nn.ConvTranspose2d(2 * ch_list[-1], ch_list[-1], 3, 2, (H_padding, W_padding), (H_padding, W_padding))
        )
        self.pe_linears_bottom = nn.Sequential(
            nn.Linear(pe_dim, ch_list[-1]), 
            nn.ReLU(),
            nn.Linear(ch_list[-1], ch_list[-1])
        )
        
    def forward(self, x, t):
        batch = x.shape[0]
        t = self.pe(t) # [batch, d_model]
        features = []
        features.append(x) # features[0] : x
        for i in range(len(self.encoder)):
            pe = self.pe_linears_en[i](t).reshape(batch, -1, 1, 1)
            features.append(self.encoder[i](features[-1] + pe))
        
        pe = self.pe_linears_bottom(t).reshape(batch, -1, 1, 1)
        x = self.bottom(features[-1] + pe)
        for i in range(len(self.decoder) - 1, -1, -1):
            pe = self.pe_linears_de[i](t).reshape(batch, -1, 1, 1)
            x = self.decoder[i](torch.cat([features[i + 1], x], dim=1) + pe)
        
        return x
```

实际测试时，发现我自己写的UNet输出的结果总是有大量的全黑或全白，而用别人的UNet就比较稳定，不知道为什么，求大佬指点一下

![image-20250502123135210](\source\image-20250502123135210.png)

## 五、应用 1：MNIST 手写数字生成

使用step=100:

![sample_step100](\source\sample_step100.png)

使用step=1000:

![sample_step1000](\source\sample_step1000.png)

## 六、应用 2：Anime Face 动漫人脸生成

数据来源：https://www.kaggle.com/datasets/soumikrakshit/anime-faces

此时训练数据集较大，采用双卡训练

主体流程如下：

```python
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
    # dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    # net = net.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 3e-4)

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
```

### **初始参数结果：**

![image-20250502134021203](\source\image-20250502134021203.png)

后面的epoch情况也差不多，loss不收敛

### 使用了另一组参数：

看了网上的其他实现，**使用了另一组参数：**（更小的模型通道数和更少的steps）

![image-20250502135406591](\source\image-20250502135406591.png)



效果反而更好一点，看来增大模型参数并不一定使效果更好

时间原因也并没有对每个变量进行对比实验

### 疑问：

每次结果都有很强的偏向性：要么整体偏蓝、要么整体偏绿、要么整体偏红

猜测是最后一层的channels到C的网络不稳定，然后迭代steps次后导致整体均值向着某个通道的方向累计偏差

解决：1.使用学习率shedueler控制学习率逐渐变小 ，2.增大最后一层的深度，3. 使用batch或layer正则化

## 七、遇到的问题及解决

### 1. **不同图像大小训练不兼容**

- **原因**：

  - 模型未使用自适应卷积（例如 kernel size 固定，图像 size 不一致）。
  - padding 或上采样时 shape 不一致导致错误。

- 解决：

  写UNet时一定要注意图像的W和H变化，每层的W、H为奇数和偶数时，反卷积的大小都会不一致，因此要分别规定好

### 2. **训练正常但生成图片模糊**

- **原因**：

  - 预测误差在多步采样中逐渐放大。
  - timestep 数过少，导致退噪不充分。

- **解决方法**：

  增加采样步数，如从 300 增加到 500。

### 3. **训练不收敛 / loss 长时间震荡**

- **原因**：
  - 噪声预测模型结构不合理或太浅。
  - 学习率过高。
  - 数据标准化或动态范围不匹配。
- **解决方法**：
  - 使用深一点的 U-Net 或 Residual U-Net。
  - 尝试 稳定的 学习率策略。
  - 图像输入归一化到 [-1, 1]；训练目标（如噪声）也应匹配相同的分布。

### 4.  **生成图像偏黑、偏白或全噪声**

- **原因**：
  - 网络太浅
  - 去噪预测不准，噪声预测器（如 U-Net）训练不充分。
  - `timestep` 编码（如时间位置编码）加入方式不合适或未加入。
- **解决方法**：
  - 增大网络深度
  - 加大训练轮数或使用更小学习率。

## 八、扩展

anime-face头像还是太单调了，后面想搞一些更多样的动漫图片。

1. 在p站上爬取几万张合适的动漫图片
2. resize到（3，128，128）或（3，256，256）
3. 然后拿来训练大一点的diffusion
4. 最后接一个超分网络（如BSRN），到（3，512，512）或1080p
5. 把结果展示到一个类似https://wangjia184.github.io/diffusion_model/的网站上

。。。

GitHub代码地址：https://github.com/PUBGprofessor/diffusion

## 参考资料

[1]https://zhouyifan.net/2023/07/07/20230330-diffusion-model/

[2]https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

[3]https://zhuanlan.zhihu.com/p/624221952

[4]https://blog.csdn.net/tobefans/article/details/129728036

[5] http://pytorch.org/tutorials/intermediate/ddp_tutorial.html

[6]https://www.cnblogs.com/qizhou/p/16770143.html

[7]https://blog.csdn.net/qq_51392112/article/details/129737803?fromshare=blogdetail&sharetype=blogdetail&sharerId=129737803&sharerefer=PC&sharesource=m0_74167177&sharefrom=from_link