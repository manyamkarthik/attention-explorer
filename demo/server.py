"""Tiny demo server: serves site/ and answers from the real PyTorch model instead of the in-browser port."""
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from attention_explorer.gpt2 import load_gpt2, run, tokenizer

SITE = Path(__file__).parent.parent / "site"
MAX_TOKENS = 40
model = load_gpt2()


@torch.no_grad()
def answer(text, causal):
    ids = tokenizer.encode(text)
    if not ids:
        return {"text": text, "causal": causal, "layers": 12, "tokens": [], "attn": [], "simple": [], "next": None, "nextMasked": None}
    trimmed = tokenizer.decode(ids[:MAX_TOKENS])
    out, masked = run(model, trimmed, causal), run(model, trimmed, True)
    x = model.tok_emb.weight[ids[:MAX_TOKENS]]
    simple = torch.softmax(x @ x.T, dim=-1)  # section 3.3, on the token embeddings
    return {
        "text": text, "causal": causal, "layers": 12, "tokens": out["tokens"], "truncated": len(ids) > MAX_TOKENS,
        "attn": out["attn"].flatten(2).tolist(), "simple": simple.flatten().tolist(),
        "next": out["next"], "nextMasked": masked["next"],
    }


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args):
        super().__init__(*args, directory=SITE)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/health":
            self.reply(b"ok", "text/plain")
        elif url.path == "/api/run":
            q = parse_qs(url.query)
            body = answer(q.get("text", [""])[0][:500], q.get("causal", ["1"])[0] == "1")
            self.reply(json.dumps(body).encode(), "application/json")
        else:
            super().do_GET()

    def reply(self, body, kind):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("http://localhost:8000")
    HTTPServer(("", 8000), Handler).serve_forever()
