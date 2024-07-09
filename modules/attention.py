import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.d_head = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, self.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, self.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.d_head, bias=False)

        self.register_buffer("tril_mask", torch.tril(torch.ones(config.context_length, config.context_length)).bool())

        self.dropout = nn.Dropout(config.p_dropout)

    def forward(self, x, attn_mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attn_scores = q @ k.transpose(-2, -1)
        attn_scores = attn_scores / self.d_head**0.5

        if x.shape[1] == self.config.context_length:
            attn_scores = attn_scores.masked_fill(~self.tril_mask, float("-inf"))
        else:
            mask = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device)).bool()
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.expand(attn_scores.shape)
            attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))

        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        z = attn_scores @ v

        return z


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.heads = nn.ModuleList([MaskedSelfAttention(config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.p_dropout)

    def forward(self, x, attn_mask=None):
        x = torch.cat([h(x, attn_mask=attn_mask) for h in self.heads], dim=-1)
        x = self.proj(x)
        x = self.dropout(x)
        return x
