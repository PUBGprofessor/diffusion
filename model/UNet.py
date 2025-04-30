import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.image import get_img_shape
from model.encoder import PositionalEncoding


# class UnetBlock(nn.Module):

#     def __init__(self, shape, in_c, out_c, residual=False):
#         super().__init__()
#         self.ln = nn.LayerNorm(shape)
#         self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
#         self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
#         self.activation = nn.ReLU()
#         self.residual = residual
#         if residual:
#             if in_c == out_c:
#                 self.residual_conv = nn.Identity()
#             else:
#                 self.residual_conv = nn.Conv2d(in_c, out_c, 1)

#     def forward(self, x):
#         out = self.ln(x)
#         out = self.conv1(out)
#         out = self.activation(out)
#         out = self.conv2(out)
#         if self.residual:
#             out += self.residual_conv(x)
#         out = self.activation(out)
#         return out

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

    def forward(self, x):
        x = self.LayerNorm(x)
        out = self.gen(x)
        if self.residual:
            out += self.residual_conv(x)
        return self.activation(out)

class MyUNet(nn.Module):
    def __init__(self, image_size, ch_list=[32, 64, 128 ,256], residual=True):
        super().__init__()
        C, H, W = image_size
        self.H = [H,]
        self.W = [W,]
        for i in range(len(ch_list) + 1): # 0, 1, 2, 3, 4
            self.H.append(self.H[-1] // 2)
            self.W.append(self.W[-1] // 2)
        # print(self.W)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # encoder最上面第一层UNetBlock
        self.encoder.append(nn.Sequential(
            UnetBlock((C, H, W), C, ch_list[0], True),
            # F.max_pool2d(),
        ))
        # decoder最上面第一层UNetBlock
        self.decoder.append(nn.Sequential(
            UnetBlock((2 * ch_list[0], H, W), 2 * ch_list[0], ch_list[0], True),
            nn.Conv2d(ch_list[0], C, 1)
        ))

        for i in range(1, len(ch_list)):

            # 此时输入为[b, ch_list[i - 1], self.H[i - 1], self.W[i - 1]]
            # 输出为[b, ch_list[i], self.H[i], self.W[i]]
            self.encoder.append(nn.Sequential(
                nn.MaxPool2d(2),
                UnetBlock((ch_list[i - 1], self.H[i], self.W[i]), ch_list[i - 1], ch_list[i], True),
                # F.max_pool2d(),
            ))

            # 此时输入为[b, 2 * ch_list[i], self.H[i], self.W[i]]
            # 输出为[b, ch_list[i-1], self.H[i - 1], self.W[i - 1]]
            H_padding = 1 if (self.H[i - 1] % 2 == 0) else 0
            W_padding = 1 if (self.W[i - 1] % 2 == 0) else 0
            self.decoder.append(nn.Sequential(
                UnetBlock((2 * ch_list[i], self.H[i], self.W[i]), 2 * ch_list[i], ch_list[i], True),
                nn.ConvTranspose2d(ch_list[i], ch_list[i - 1], 3, 2, 1, (H_padding, W_padding))
            ))

        # 输入 [b, ch_list[-1], self.H[-2], self.W[-2]]
        # 输出 [b, ch_list[-1], self.H[-2], self.W[-2]]
        H_padding = 1 if (self.H[-2] % 2 == 0) else 0
        W_padding = 1 if (self.W[-2] % 2 == 0) else 0
        self.bottom = nn.Sequential(
            nn.MaxPool2d(2),
            UnetBlock((ch_list[-1], self.H[-2], self.W[-2]), ch_list[-1], 2 * ch_list[-1], True),
            nn.ConvTranspose2d(2 * ch_list[-1], ch_list[-1], 3, 2, 1, (H_padding, W_padding))
        )
        
    def forward(self, x):
        features = []
        features.append(x) # features[0] : x
        for i in range(len(self.encoder)):
            features.append(self.encoder[i](features[-1]))
        
        x = self.bottom(features[-1])
        for i in range(len(self.decoder) - 1, -1, -1):
            x = self.decoder[i](torch.cat([features[i + 1], x], dim=1))
        
        return x

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
            nn.Conv2d(ch_list[0], C, 1)
        ))
        self.pe_linears_de.append(nn.Sequential(
            nn.Linear(pe_dim, 2 * ch_list[0]), 
            nn.ReLU(),
            nn.Linear(2 * ch_list[0], 2 * ch_list[0])
        ))

        for i in range(1, len(ch_list)):

            # 此时输入为[b, ch_list[i - 1], self.H[i - 1], self.W[i - 1]]
            # 输出为[b, ch_list[i], self.H[i], self.W[i]]
            self.encoder.append(nn.Sequential(
                nn.MaxPool2d(2),
                UnetBlock((ch_list[i - 1], self.H[i], self.W[i]), ch_list[i - 1], ch_list[i], True),
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
                nn.ConvTranspose2d(ch_list[i], ch_list[i - 1], 3, 2, 1, (H_padding, W_padding))
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
            nn.ConvTranspose2d(2 * ch_list[-1], ch_list[-1], 3, 2, 1, (H_padding, W_padding))
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

class UNet(nn.Module):
    def __init__(self,
                 n_steps,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 residual=False) -> None:
        super().__init__()
        C, H, W = get_img_shape()
        layers = len(channels) # 4
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers - 1): # 0,1,2,3
            cH //= 2
            cW //= 2
            Hs.append(cH)
            Ws.append(cW)

        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_channel = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]): # 0,1,2,3
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))
            self.encoders.append(
                nn.Sequential(
                    UnetBlock((prev_channel, cH, cW),
                              prev_channel,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        channel = channels[-1]
        self.mid = nn.Sequential(
            UnetBlock((prev_channel, Hs[-1], Ws[-1]),
                      prev_channel,
                      channel,
                      residual=residual),
            UnetBlock((channel, Hs[-1], Ws[-1]),
                      channel,
                      channel,
                      residual=residual),
        )
        prev_channel = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))
            self.decoders.append(
                nn.Sequential(
                    UnetBlock((channel * 2, cH, cW),
                              channel * 2,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))

            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t) # [batch, d_model]
        encoder_outs = []
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders,
                                            self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)
        pe = self.pe_mid(t).reshape(n, -1, 1, 1)
        x = self.mid(x + pe)
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = up(x)

            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x + pe)
        x = self.conv_out(x)
        return x