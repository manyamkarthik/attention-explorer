"""Step 3 (3.5): causal attention. Hide the future so token t can only look at tokens 0..t, plus dropout."""
import torch
import torch.nn as nn

from .simplified import INPUTS, WORDS


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.causal = True  # flip off to let the model peek at the future

    def forward(self, x, return_weights=False):
        b, num_tokens, d_in = x.shape
        queries, keys, values = self.W_query(x), self.W_key(x), self.W_value(x)
        scores = queries @ keys.transpose(1, 2)
        if self.causal:
            scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # -inf -> 0 after softmax
        weights = self.dropout(torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1))
        return (weights @ values, weights) if return_weights else weights @ values


def peek_test(attn, x):
    """Change only the LAST token. If any earlier output moves, earlier tokens were reading the future."""
    changed = x.clone()
    changed[:, -1] += 10.0
    with torch.no_grad():
        moved = (attn(x) - attn(changed)).abs().sum(-1)[0, :-1]
    return moved


if __name__ == "__main__":
    torch.manual_seed(123)
    T = INPUTS.shape[0]
    print("the mask (1 = future, gets -inf):")
    print(torch.triu(torch.ones(T, T), diagonal=1).int())

    batch = torch.stack((INPUTS, INPUTS))  # (2, 6, 3)
    ca = CausalAttention(3, 2, context_length=T, dropout=0.0)
    _, w = ca(batch, return_weights=True)
    print("\ncausal weights (upper triangle is exactly 0, rows still sum to 1):")
    print(w[0].detach().numpy().round(3))

    print("\ndropout 0.5 in training mode: half the weights zeroed, the rest scaled x2")
    torch.manual_seed(123)
    print(nn.Dropout(0.5)(torch.ones(T, T)))

    print("\nwhat breaks without the mask: change only the last word ('step') and see who notices")
    x = INPUTS.unsqueeze(0)
    for causal in (True, False):
        ca.causal = causal
        moved = peek_test(ca, x)
        print(f"  causal={str(causal):5}  output change for {WORDS[:-1]}: {moved.numpy().round(3)}")
    print("Without the mask every earlier token's output depends on a word it hasn't 'seen' yet.\n"
          "In training that's cheating: predicting the next word is trivial when you can read it.")
