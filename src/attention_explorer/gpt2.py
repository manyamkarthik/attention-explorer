"""Step 5: the MultiHeadAttention from this chapter, with GPT-2 small's trained weights plugged in.

Random weights give random heatmaps. To see heads that actually track meaning ("it" -> "animal"),
we need a trained model, so this wraps our attention class in the minimum GPT-2 around it
(layer norm + feed-forward, from chapter 4) and loads OpenAI's weights from Hugging Face.
"""
import sys

import tiktoken
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .multihead import MultiHeadAttention

CFG = {"vocab_size": 50257, "context_length": 1024, "emb_dim": 768, "n_heads": 12, "n_layers": 12}


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.att = MultiHeadAttention(d, d, cfg["context_length"], 0.0, cfg["n_heads"], qkv_bias=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT2(nn.Module):
    def __init__(self, cfg=CFG):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layers"])])
        self.ln_f = nn.LayerNorm(cfg["emb_dim"])

    def forward(self, ids):
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(ids.shape[1]))
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x) @ self.tok_emb.weight.T  # GPT-2 ties the output layer to the token embedding

    def set_causal(self, causal):
        for blk in self.blocks:
            blk.att.causal = causal


def hf_weights():
    """GPT-2 small from Hugging Face as {name: tensor}. Conv1D weights are (in, out), so they get transposed later."""
    sd = load_file(hf_hub_download("openai-community/gpt2", "model.safetensors"))
    return {k.removeprefix("transformer."): v for k, v in sd.items()}


def load_gpt2(sd=None):
    sd, model = sd or hf_weights(), GPT2()
    copy = lambda p, t: p.data.copy_(t)
    copy(model.tok_emb.weight, sd["wte.weight"])
    copy(model.pos_emb.weight, sd["wpe.weight"])
    copy(model.ln_f.weight, sd["ln_f.weight"])
    copy(model.ln_f.bias, sd["ln_f.bias"])
    for i, blk in enumerate(model.blocks):
        h = lambda name: sd[f"h.{i}.{name}"]
        q, k, v = h("attn.c_attn.weight").T.chunk(3, dim=0)  # one fused (768, 2304) matrix -> three (768, 768)
        qb, kb, vb = h("attn.c_attn.bias").chunk(3)
        for lin, w, b in ((blk.att.W_query, q, qb), (blk.att.W_key, k, kb), (blk.att.W_value, v, vb)):
            copy(lin.weight, w)
            copy(lin.bias, b)
        copy(blk.att.out_proj.weight, h("attn.c_proj.weight").T)
        copy(blk.att.out_proj.bias, h("attn.c_proj.bias"))
        copy(blk.ff[0].weight, h("mlp.c_fc.weight").T)
        copy(blk.ff[0].bias, h("mlp.c_fc.bias"))
        copy(blk.ff[2].weight, h("mlp.c_proj.weight").T)
        copy(blk.ff[2].bias, h("mlp.c_proj.bias"))
        copy(blk.ln1.weight, h("ln_1.weight"))
        copy(blk.ln1.bias, h("ln_1.bias"))
        copy(blk.ln2.weight, h("ln_2.weight"))
        copy(blk.ln2.bias, h("ln_2.bias"))
    return model.eval()


tokenizer = tiktoken.get_encoding("gpt2")


@torch.no_grad()
def run(model, text, causal=True):
    """Tokens, attention weights for every layer and head (L, H, T, T), and the top next-token guess at each position."""
    ids = tokenizer.encode(text)
    model.set_causal(causal)
    logits = model(torch.tensor([ids]))
    model.set_causal(True)
    return {
        "tokens": [tokenizer.decode([i]) for i in ids],
        "attn": torch.stack([blk.att.last_weights[0] for blk in model.blocks]),
        "next": [tokenizer.decode([i]) for i in logits[0].argmax(-1).tolist()],
    }


def strongest_head(attn, tokens, src, dst):
    """Which (layer, head) puts the most weight from token `src` onto token `dst`?"""
    i = next(n for n, t in enumerate(tokens) if t.strip() == src)
    j = next(n for n, t in enumerate(tokens) if t.strip() == dst)
    w = attn[:, :, i, j]
    layer, head = divmod(int(w.argmax()), w.shape[1])
    return layer, head, float(w.max())


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "The animal didn't cross the street because it was too tired"
    model = load_gpt2()
    out = run(model, text)
    print(f"tokens : {out['tokens']}")
    print(f"attn   : {tuple(out['attn'].shape)}  (layers, heads, query, key)")
    print(f"next   : {out['next']}")
    if "it" in [t.strip() for t in out["tokens"]] and "animal" in [t.strip() for t in out["tokens"]]:
        L, H, w = strongest_head(out["attn"], out["tokens"], "it", "animal")
        print(f"\nstrongest 'it' -> 'animal' head: layer {L}, head {H}  ({w:.0%} of its attention)")
