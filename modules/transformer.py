import math

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention import MultiHeadAttention
from modules.positional_encoding import PositionalEncoding


@dataclass
class TransformerConfig:
    vocab_size: int = 15256
    d_model: int = 768
    context_length: int = 512
    n_heads: int = 12
    n_layers: int = 12
    p_dropout: float = 0.1


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
    def __init__(self, config=TransformerConfig()):
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

        # initialize parameters so that activations would have same std across layers
        for i, block in enumerate(self.blocks, start=1):
            std = 0.02 / math.sqrt(2 * i)
            for module in block.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                    if module.bias is not None:
                        torch.nn.init.constant_(module.bias, 0)
        torch.nn.init.normal_(self.token_embedding_table.weight, mean=0.0, std=0.02)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, attn_mask=None):
        x = self.token_embedding_table(x)
        x = self.positional_encoding(x)

        for i, block in enumerate(self.blocks):
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

    def save(self, path):
        pkg = {
            "config": self.config,
            "model_state_dict": self.state_dict(),
        }
        torch.save(pkg, path)

    @classmethod
    def init_and_load(cls, path):
        pkg = torch.load(path, map_location="cpu")
        model = cls(pkg["config"])
        model.load_state_dict(pkg["model_state_dict"])
        return model

    @torch.no_grad()
    def generate(self, input_ids, new_tokens_number, attn_mask=None, **kwargs):
        self.eval()

        attn_mask = attn_mask.to(self.device) if attn_mask is not None else None
        output = input_ids.to(self.device)

        for _ in range(new_tokens_number):
            logits = self(output, attn_mask=attn_mask)
            logits = logits[:, -1, :]
            new_tokens = self._sample(logits, generated_tokens=output, **kwargs)
            output = torch.cat([output, new_tokens], dim=1)

            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, torch.ones_like(new_tokens, dtype=torch.bool)], dim=1)

        return output.cpu()


    def _sample(
        self,
        logits,
        generated_tokens,
        greedy=False,
        temperature=1.0,
        top_k=50,
        top_p=None,
        repetition_penalty=1.0,
    ):
        if greedy:
            return torch.argmax(logits, dim=-1, keepdim=True)
        else:

            # Check if both top_k and top_p are specified
            if top_k is not None and top_p is not None:
                raise ValueError("Both top_k and top_p cannot be used simultaneously.")

            # Apply top-k filtering if specified
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p filtering if specified
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p

                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Replace logits to be removed with -inf in the sorted_logits
                sorted_logits[sorted_indices_to_remove] = float("-inf")

                # Then reverse the sorting process by mapping back sorted_logits to their original position
                logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

            # Apply repetition_penalty if not default
            if repetition_penalty != 1.0:
                for i in range(logits.shape[0]):
                    sequence_tokens = generated_tokens[i]
                    for token_id in sequence_tokens:
                        if token_id > 0:  # Ignore padding
                            logits[i, token_id] /= repetition_penalty

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, 1)
            return ix
