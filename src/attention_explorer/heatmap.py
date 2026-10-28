"""Step 6: see it. Heatmaps of the attention weights, one head or a whole layer of heads.

uv run python -m attention_explorer.heatmap "The animal didn't cross the street because it was too tired" --layer 4 --head 3
uv run python -m attention_explorer.heatmap "..." --layer 4              # all 12 heads of layer 4
uv run python -m attention_explorer.heatmap "..." --layer 4 --no-mask    # what the model sees when it can peek
"""
import argparse

import matplotlib.pyplot as plt

from .gpt2 import load_gpt2, run


def draw(ax, w, tokens, title):
    ax.imshow(w, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)), tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tokens)), tokens, fontsize=8)
    ax.set_title(title, fontsize=9)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", default="The animal didn't cross the street because it was too tired")
    p.add_argument("--layer", type=int, default=4)
    p.add_argument("--head", type=int)
    p.add_argument("--no-mask", action="store_true")
    p.add_argument("--out", default="attention.png")
    a = p.parse_args()

    out = run(load_gpt2(), a.text, causal=not a.no_mask)
    tokens, attn = [t.replace("\n", "\\n") for t in out["tokens"]], out["attn"][a.layer]
    if a.head is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        draw(ax, attn[a.head], tokens, f"layer {a.layer}, head {a.head}  (rows = query, cols = key)")
    else:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for h, ax in enumerate(axes.flat):
            draw(ax, attn[h], tokens, f"head {h}")
        fig.suptitle(f"layer {a.layer}, all heads{'  (NO causal mask)' if a.no_mask else ''}")
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"saved {a.out}")
