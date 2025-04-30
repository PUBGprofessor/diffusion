import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import get_img_shape
from model.encoder import PositionalEncoding

class ResidualBlock(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.actvation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.actvation2 = nn.ReLU()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                          nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.actvation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(input)
        x = self.actvation2(x)
        return x
    
class ConvNet(nn.Module):

    def __init__(self,
                 n_steps,
                 intermediate_channels=[10, 20, 40],
                 pe_dim=10,
                 insert_t_to_all_layers=False):
        super().__init__()
        C, H, W = get_img_shape()  # 1, 28, 28
        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.pe_linears = nn.ModuleList()
        self.all_t = insert_t_to_all_layers
        if not insert_t_to_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim, C))

        self.residual_blocks = nn.ModuleList()
        prev_channel = C
        for channel in intermediate_channels:
            self.residual_blocks.append(ResidualBlock(prev_channel, channel))
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))
            else:
                self.pe_linears.append(None)
            prev_channel = channel
        self.output_layer = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        for m_x, m_t in zip(self.residual_blocks, self.pe_linears):
            if m_t is not None:
                pe = m_t(t).reshape(n, -1, 1, 1)
                x = x + pe
            x = m_x(x)
        x = self.output_layer(x)
        return x




# convnet_small_cfg = {
#     'type': 'ConvNet',
#     'intermediate_channels': [10, 20],
#     'pe_dim': 128
# }

# convnet_medium_cfg = {
#     'type': 'ConvNet',
#     'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],
#     'pe_dim': 256,
#     'insert_t_to_all_layers': True
# }
# convnet_big_cfg = {
#     'type': 'ConvNet',
#     'intermediate_channels': [20, 20, 40, 40, 80, 80, 160, 160],
#     'pe_dim': 256,
#     'insert_t_to_all_layers': True
# }

# unet_1_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
# unet_res_cfg = {
#     'type': 'UNet',
#     'channels': [10, 20, 40, 80],
#     'pe_dim': 128,
#     'residual': True
# }


# def build_network(config: dict, n_steps):
#     network_type = config.pop('type')
#     if network_type == 'ConvNet':
#         network_cls = ConvNet
#     elif network_type == 'UNet':
#         network_cls = UNet

#     network = network_cls(n_steps, **config)
#     return network