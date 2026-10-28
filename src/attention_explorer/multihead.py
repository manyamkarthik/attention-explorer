"""Step 4 (3.6): multi-head attention. First stack separate heads, then do it properly with one matmul and a reshape."""
import torch
import torch.nn as nn

from .causal import CausalAttention
from .simplified import INPUTS


class MultiHeadAttentionWrapper(nn.Module):
    """num_heads independent CausalAttention modules, run one after another, outputs concatenated."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    """One big projection, split into heads by reshaping, so every head runs in the same matmul."""

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out, self.num_heads, self.head_dim = d_out, num_heads, d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # mixes the heads back together
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.causal = True
        self.last_weights = None  # (b, heads, T, T), kept for the heatmaps

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        split = lambda t: t.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)  # (b, heads, T, head_dim)
        queries, keys, values = split(self.W_query(x)), split(self.W_key(x)), split(self.W_value(x))

        scores = queries @ keys.transpose(2, 3)  # (b, heads, T, T): every head at once
        if self.causal:
            scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        weights = torch.softmax(scores / self.head_dim ** 0.5, dim=-1)
        self.last_weights = weights.detach()

        context = (self.dropout(weights) @ values).transpose(1, 2)  # (b, T, heads, head_dim)
        return self.out_proj(context.contiguous().view(b, num_tokens, self.d_out))  # concat heads


if __name__ == "__main__":
    torch.manual_seed(123)
    batch = torch.stack((INPUTS, INPUTS))
    b, T, d_in = batch.shape

    wrap = MultiHeadAttentionWrapper(d_in, 2, T, 0.0, num_heads=2)
    print(f"wrapper : 2 heads x d_out=2, concatenated -> {tuple(wrap(batch).shape)}")

    mha = MultiHeadAttention(d_in, 4, T, 0.0, num_heads=2)
    out = mha(batch)
    print(f"MHA     : d_out=4 split into 2 heads of {mha.head_dim} -> {tuple(out.shape)}")
    print(f"weights : {tuple(mha.last_weights.shape)}  (batch, heads, query, key)")
    for h in range(2):
        print(f"\nhead {h}:")
        print(mha.last_weights[0, h].numpy().round(3))
    print("\nTwo heads, two different attention patterns over the same tokens. GPT-2 small runs 12 of these per layer.")
