// GPT-2 small in plain JavaScript: the tokenizer from tokenizer-lab, then the same forward pass as
// src/attention_explorer/gpt2.py, reading the int8 weights from scripts/export_weights.py. Runs in a worker or in node.
const EOT = "<|endoftext|>";
const WORD = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
const utf8 = new TextEncoder(), show8 = new TextDecoder();

// ---- GPT-2 BPE tokenizer (copied from tokenizer-lab/site/lab.js)
function byteMaps() {
  const bs = [...range(33, 126), ...range(161, 172), ...range(174, 255)], cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
  const enc = {}, dec = {};
  bs.forEach((b, i) => { enc[b] = String.fromCharCode(cs[i]); dec[String.fromCharCode(cs[i])] = b; });
  return [enc, dec];
}
function* range(a, b) { for (let i = a; i <= b; i++) yield i; }

class GPT2 {
  constructor(encoder, bpeText) {
    this.encoder = encoder;
    this.ranks = new Map(bpeText.split("\n").slice(1).filter(Boolean).map((l, i) => [l, i]));
    [this.b2u, this.u2b] = byteMaps();
    this.id2tok = Object.fromEntries(Object.entries(encoder).map(([t, i]) => [i, t]));
    this.size = 50257;
  }
  bpe(word) {
    let sym = [...word];
    while (sym.length > 1) {
      let best = -1, rank = Infinity;
      for (let i = 0; i < sym.length - 1; i++) {
        const r = this.ranks.get(sym[i] + " " + sym[i + 1]);
        if (r !== undefined && r < rank) { rank = r; best = i; }
      }
      if (best < 0) break;
      const [a, b] = [sym[best], sym[best + 1]], out = [];
      for (let i = 0; i < sym.length; i++) {
        if (i < sym.length - 1 && sym[i] === a && sym[i + 1] === b) { out.push(a + b); i++; } else out.push(sym[i]);
      }
      sym = out;
    }
    return sym;
  }
  pieces(text) {
    const out = [];
    for (const chunk of text.split(EOT).flatMap((c, i, a) => (i < a.length - 1 ? [c, EOT] : [c]))) {
      if (chunk === EOT) { out.push([EOT, 50256]); continue; }
      for (const w of chunk.match(WORD) || []) {
        const mapped = Array.from(utf8.encode(w), b => this.b2u[b]).join("");
        for (const s of this.bpe(mapped)) out.push([s, this.encoder[s]]);
      }
    }
    return out;
  }
  encode(text) { return this.pieces(text).map(p => p[1]); }
  tokens(text) {
    return this.pieces(text).map(([s, id]) =>
      id === 50256 ? EOT : show8.decode(Uint8Array.from([...s], c => this.u2b[c])));
  }
}

// ---- weights: int8 rows with one fp32 scale each, token embedding in fp16 (see scripts/export_weights.py)
const F16 = new Float32Array(65536);  // every fp16 bit pattern -> its float value
for (let h = 0; h < 65536; h++) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  F16[h] = s * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15));
}

class Weights {
  constructor(manifest, fetchBuf) {
    this.m = manifest; this.fetchBuf = fetchBuf; this.files = {}; this.layers = 0; this.wte = []; this.L = [];
  }
  async file(name) {
    return (this.files[name] ??= this.fetchBuf("weights/" + name));
  }
  async t(name) {
    const e = this.m.tensors[name], buf = await this.file(e.file), n = e.shape.reduce((a, b) => a * b);
    if (e.dtype === "f32") return new Float32Array(buf, e.offset, n);
    if (e.dtype === "f16") return new Uint16Array(buf, e.offset, n);
    return { q: new Int8Array(buf, e.offset, n), s: new Float32Array(buf, e.scale, e.shape[0]), out: e.shape[0], in: e.shape[1] };
  }
  async loadBase() {
    this.wpe = await this.t("wpe"); this.lnf = [await this.t("ln_f.w"), await this.t("ln_f.b")];
  }
  async loadLayer(i) {
    const g = n => this.t(`${i}.${n}`);
    this.L[i] = {
      ln1: [await g("ln_1.w"), await g("ln_1.b")], ln2: [await g("ln_2.w"), await g("ln_2.b")],
      attn: [await g("attn.c_attn.w"), await g("attn.c_attn.b")], proj: [await g("attn.c_proj.w"), await g("attn.c_proj.b")],
      fc: [await g("mlp.c_fc.w"), await g("mlp.c_fc.b")], fcproj: [await g("mlp.c_proj.w"), await g("mlp.c_proj.b")],
    };
    this.layers = i + 1;
  }
  shardOf(id) { return Math.floor(id / this.m.wte_shard); }
  async loadShard(k) { return (this.wte[k] ??= await this.t("wte" + k)); }
  get allShards() { return Math.ceil(50257 / this.m.wte_shard); }
  hasAllShards() { for (let k = 0; k < this.allShards; k++) if (!this.wte[k]) return false; return true; }
  tokRow(id) {
    const sh = this.wte[this.shardOf(id)], o = (id % this.m.wte_shard) * 768, r = new Float32Array(768);
    for (let i = 0; i < 768; i++) r[i] = F16[sh[o + i]];
    return r;
  }
}

// ---- the forward pass, one Float32Array of 768 per token
const D = 768, H = 12, HD = 64;

function layerNorm(x, [w, b]) {
  let mu = 0, v = 0;
  for (let i = 0; i < D; i++) mu += x[i];
  mu /= D;
  for (let i = 0; i < D; i++) v += (x[i] - mu) ** 2;
  const r = 1 / Math.sqrt(v / D + 1e-5), y = new Float32Array(D);
  for (let i = 0; i < D; i++) y[i] = (x[i] - mu) * r * w[i] + b[i];
  return y;
}

// y = W x + b for every token at once, so each int8 row is read once per call
function linear(xs, [W, b]) {
  const ys = xs.map(() => new Float32Array(W.out));
  for (let o = 0; o < W.out; o++) {
    const row = o * W.in, s = W.s[o];
    for (let t = 0; t < xs.length; t++) {
      const x = xs[t];
      let acc = 0;
      for (let i = 0; i < W.in; i++) acc += W.q[row + i] * x[i];
      ys[t][o] = acc * s + b[o];
    }
  }
  return ys;
}

const gelu = x => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));

function softmax(a) {
  const m = Math.max(...a);
  let z = 0;
  for (let i = 0; i < a.length; i++) z += a[i] = a[i] === -Infinity ? 0 : Math.exp(a[i] - m);
  for (let i = 0; i < a.length; i++) a[i] /= z;
  return a;
}

// Chapter 3's MultiHeadAttention: split into 12 heads of 64, scaled dot product, causal mask, concat
function attention(hs, layer, causal) {
  const T = hs.length, qkv = linear(hs, layer.attn), weights = [], ctx = hs.map(() => new Float32Array(D));
  for (let h = 0; h < H; h++) {
    const w = new Float32Array(T * T), q = h * HD, k = D + h * HD, v = 2 * D + h * HD;
    for (let i = 0; i < T; i++) {
      const row = new Array(T);
      for (let j = 0; j < T; j++) {
        if (causal && j > i) { row[j] = -Infinity; continue; }
        let s = 0;
        for (let d = 0; d < HD; d++) s += qkv[i][q + d] * qkv[j][k + d];
        row[j] = s / 8;  // sqrt(head_dim)
      }
      softmax(row);
      for (let j = 0; j < T; j++) {
        w[i * T + j] = row[j];
        if (row[j]) for (let d = 0; d < HD; d++) ctx[i][h * HD + d] += row[j] * qkv[j][v + d];
      }
    }
    weights.push(w);
  }
  return [linear(ctx, layer.proj), weights];
}

// ids -> attention weights [layer][head] (T*T each) for every loaded layer, plus next-token guesses once all is loaded
function forward(W, ids, causal = true) {
  let xs = ids.map((id, t) => { const r = W.tokRow(id); for (let i = 0; i < D; i++) r[i] += W.wpe[t * D + i]; return r; });
  const attn = [];
  for (let l = 0; l < W.layers; l++) {
    const layer = W.L[l], [a, w] = attention(xs.map(x => layerNorm(x, layer.ln1)), layer, causal);
    attn.push(w);
    xs = xs.map((x, t) => x.map((v, i) => v + a[t][i]));
    const f = linear(linear(xs.map(x => layerNorm(x, layer.ln2)), layer.fc).map(h => h.map(gelu)), layer.fcproj);
    xs = xs.map((x, t) => x.map((v, i) => v + f[t][i]));
  }
  let next = null;
  if (W.layers === 12 && W.hasAllShards()) {
    next = xs.map(x => {
      const y = layerNorm(x, W.lnf);
      let best = 0, top = -Infinity;
      for (let k = 0; k < W.allShards; k++) {
        const sh = W.wte[k], n = sh.length / D;
        for (let r = 0; r < n; r++) {
          let s = 0;
          const o = r * D;
          for (let i = 0; i < D; i++) s += F16[sh[o + i]] * y[i];
          if (s > top) { top = s; best = k * W.m.wte_shard + r; }
        }
      }
      return best;
    });
  }
  return { attn, next };
}

// Chapter 3.3: no weights at all, just softmax(x @ x.T) over the token embeddings
function simplified(W, ids) {
  const xs = ids.map(id => W.tokRow(id)), T = ids.length, w = new Float32Array(T * T);
  for (let i = 0; i < T; i++) {
    const row = xs.map(x => { let s = 0; for (let d = 0; d < D; d++) s += xs[i][d] * x[d]; return s; });
    softmax(row).forEach((v, j) => (w[i * T + j] = v));
  }
  return w;
}

if (typeof module !== "undefined") module.exports = { GPT2, Weights, forward, simplified, EOT };
