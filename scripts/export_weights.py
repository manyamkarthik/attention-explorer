"""Shrink GPT-2 small for the browser: int8 with one fp32 scale per row, split so the page fetches only what it needs.

uv run python scripts/export_weights.py            # writes site/weights/ (~160 MB, git-ignored; CI builds it)

Every matrix is stored as (out, in) so the JS forward pass is a row-by-row dot product.
The token embedding stays fp16: it doubles as the output layer, and int8 there flips the next-token guesses.
"""
import json
from pathlib import Path

import numpy as np
import torch

from attention_explorer.gpt2 import hf_weights, load_gpt2, run

OUT = Path(__file__).parent.parent / "site" / "weights"
MAX_TOKENS = 128   # the page caps input length, so only the first 128 position embeddings ship
WTE_SHARD = 4096   # token embedding rows per file


def quantize(w):
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
    return torch.round(w / scale).to(torch.int8), scale.squeeze(1)


class Writer:
    def __init__(self):
        self.manifest, self.files = {}, {}

    def add(self, file, name, t, dtype=None):
        buf = self.files.setdefault(file, bytearray())
        dtype = dtype or ("i8" if t.dim() == 2 else "f32")
        buf += b"\0" * (-len(buf) % 4)  # keep Float32Array views aligned
        self.manifest[name] = {"file": file, "offset": len(buf), "shape": list(t.shape), "dtype": dtype}
        if dtype == "i8":
            q, scale = quantize(t.float())
            buf += q.numpy().tobytes()
            buf += b"\0" * (-len(buf) % 4)
            self.manifest[name]["scale"] = len(buf)
            buf += scale.numpy().astype(np.float32).tobytes()
        else:
            buf += t.float().numpy().astype(np.float16 if dtype == "f16" else np.float32).tobytes()

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        for f, buf in self.files.items():
            (OUT / f).write_bytes(buf)
        (OUT / "manifest.json").write_text(json.dumps({"tensors": self.manifest, "wte_shard": WTE_SHARD, "max_tokens": MAX_TOKENS}))
        print(f"wrote {len(self.files)} files, {sum(map(len, self.files.values())) / 1e6:.0f} MB -> {OUT}")


def dequantize(sd):
    """The state dict as the browser sees it: every matrix rounded through int8 exactly the way it's exported."""
    def roundtrip(w):
        q, s = quantize(w.float())
        return q.float() * s[:, None]

    conv1d = lambda k: k.endswith(("c_attn.weight", "c_proj.weight", "c_fc.weight"))
    out = {k: (roundtrip(v.T).T if conv1d(k) else roundtrip(v)) if v.dim() == 2 else v for k, v in sd.items()}
    out["wte.weight"], out["wpe.weight"] = sd["wte.weight"].half().float(), sd["wpe.weight"]
    return out


if __name__ == "__main__":
    sd = hf_weights()
    w = Writer()
    w.add("embed.bin", "wpe", sd["wpe.weight"][:MAX_TOKENS], "f32")
    w.add("embed.bin", "ln_f.w", sd["ln_f.weight"])
    w.add("embed.bin", "ln_f.b", sd["ln_f.bias"])
    for s in range(0, sd["wte.weight"].shape[0], WTE_SHARD):
        w.add(f"wte{s // WTE_SHARD}.bin", f"wte{s // WTE_SHARD}", sd["wte.weight"][s:s + WTE_SHARD], "f16")
    for i in range(12):
        h, f = lambda n: sd[f"h.{i}.{n}"], f"layer{i}.bin"
        for name in ("ln_1", "ln_2"):
            w.add(f, f"{i}.{name}.w", h(f"{name}.weight"))
            w.add(f, f"{i}.{name}.b", h(f"{name}.bias"))
        for name in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
            w.add(f, f"{i}.{name}.w", h(f"{name}.weight").T)  # Conv1D (in, out) -> (out, in)
            w.add(f, f"{i}.{name}.b", h(f"{name}.bias"))
    w.save()

    # does int8 still show the same attention? compare against the full-precision model
    text = "The animal didn't cross the street because it was too tired"
    ref, q = run(load_gpt2(sd), text), run(load_gpt2(dequantize(sd)), text)
    print(f"int8 vs fp32: max attention diff {(ref['attn'] - q['attn']).abs().max():.4f}, "
          f"next-token guesses that differ: {sum(a != b for a, b in zip(ref['next'], q['next']))}/{len(q['next'])}")
    (OUT / "reference.json").write_text(json.dumps({"text": text, "tokens": q["tokens"], "next": q["next"],
                                                    "attn_4_3": q["attn"][4, 3].tolist()}))
