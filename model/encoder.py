import torch.nn as nn
import torch

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