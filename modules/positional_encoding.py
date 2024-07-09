import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, context_length, n=10_000):
        super().__init__()

        position = torch.arange(context_length).unsqueeze(1)
        denominator = torch.exp(torch.arange(0, d_model, 2) * (-math.log(n) / d_model))
        pe = torch.zeros(context_length, d_model)
        pe[:, 0::2] = torch.sin(position * denominator)
        pe[:, 1::2] = torch.cos(position * denominator)
        self.register_buffer("pe", pe)

    def forward(self, x):
        B, T, C= x.shape

        x = x + self.pe[torch.arange(T, device=x.device)]
        return x
