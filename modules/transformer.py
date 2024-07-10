from dataclasses import dataclass

import torch
import torch.nn as nn

from modules.attention import MultiHeadAttention
from modules.positional_encoding import PositionalEncoding


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int
    context_length: int
    n_heads: int
    n_layers: int
    p_dropout: float


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn = MultiHeadAttention(config)
        self.ffn = nn.Sequential(
            *[
                nn.Linear(config.d_model, config.d_model * 4),
                nn.GELU(),
                nn.Linear(config.d_model * 4, config.d_model),
                nn.Dropout(config.p_dropout),
            ]
        )
        self.ln1 = nn.LayerNorm(config.d_model)
        self.ln2 = nn.LayerNorm(config.d_model)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding_table = torch.nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.context_length)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.out_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

        # weight sharing
        self.token_embedding_table.weight = self.out_projection.weight

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, attn_mask=None):
        x = self.token_embedding_table(x)
        x = self.positional_encoding(x)

        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln(x)
        x = self.out_projection(x)
        return x

    def compute_loss(self, x, y, attn_mask=None):
        logits = self(x, attn_mask=attn_mask)
        logits_flat = logits.view(-1, logits.shape[-1])

        y_flat = y.view(-1)

        loss = self.loss_fn(logits_flat, y_flat)

        if attn_mask is not None:
            attn_mask_flat = attn_mask.view(-1)
            loss = loss.masked_select(attn_mask_flat)

        return loss.mean()

    def generate(self, prompt, max_length):
        pass

    def save(self, path):
        pkg = dict(
            config=self.config,
            state_dict=self.state_dict(),
        )
        torch.save(pkg, path)

    @classmethod
    def init_and_load(cls, path):
        pkg = torch.load(path)
        model = cls(pkg["config"])
        model.load_state_dict(pkg["state_dict"])
        return model
