"""Step 2 (3.4): the trainable version. Project inputs into queries, keys and values, scale by sqrt(d_k)."""
import torch
import torch.nn as nn

from .simplified import INPUTS, simplified_attention


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        queries, keys, values = x @ self.W_query, x @ self.W_key, x @ self.W_value
        scores = queries @ keys.T
        weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)
        return weights @ values


class SelfAttention_v2(nn.Module):
    """Same math, but nn.Linear gives a better weight init (and an optional bias)."""

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x, return_weights=False):
        queries, keys, values = self.W_query(x), self.W_key(x), self.W_value(x)
        scores = queries @ keys.transpose(-2, -1)
        weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)
        return (weights @ values, weights) if return_weights else weights @ values


if __name__ == "__main__":
    torch.manual_seed(789)
    d_in, d_out = INPUTS.shape[1], 2

    _, ctx0 = simplified_attention(INPUTS)
    v1, v2 = SelfAttention_v1(d_in, d_out), SelfAttention_v2(d_in, d_out)
    print(f"simplified   : context {tuple(ctx0.shape)}   <- no weights, output size = input size")
    print(f"v1 (Param)   : context {tuple(v1(INPUTS).shape)}   <- W_q, W_k, W_v are {d_in}x{d_out} matrices")
    print(f"v2 (Linear)  : context {tuple(v2(INPUTS).shape)}")

    # nn.Linear stores W transposed, so copying v2's weights into v1 must give identical outputs
    with torch.no_grad():
        v1.W_query.copy_(v2.W_query.weight.T)
        v1.W_key.copy_(v2.W_key.weight.T)
        v1.W_value.copy_(v2.W_value.weight.T)
    print(f"\nv1 with v2's weights == v2 ? {torch.allclose(v1(INPUTS), v2(INPUTS))}")
    print(v2(INPUTS).detach().numpy().round(4))
