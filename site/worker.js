// Runs model.js off the main thread: streams the weights in layer by layer and re-answers the latest request as they land.
importScripts("model.js");

const MAX_TOKENS = 40;
const fetchBuf = async f => {
  const r = await fetch(f);
  if (!r.ok) throw new Error(`${f}: ${r.status}`);
  return r.arrayBuffer();
};
let W, tk, latest = null, running = false, again = false, bytes = 0;
const masked = new Map();  // text -> next-token guesses with the mask on, to compare against peeking

const decode = id => show8.decode(Uint8Array.from([...tk.id2tok[id]], c => tk.u2b[c]));

async function answer() {
  if (running) { again = true; return; }
  if (!latest || !W) return;
  running = true;
  const { text, causal } = latest;
  const ids = tk.encode(text).slice(0, MAX_TOKENS);
  for (const id of ids) await W.loadShard(W.shardOf(id));
  const out = ids.length ? forward(W, ids, causal) : { attn: [], next: null };
  let nextMasked = null;
  if (out.next) {
    if (causal) masked.set(text, out.next);
    else if (!masked.has(text)) masked.set(text, forward(W, ids, true).next);
    nextMasked = masked.get(text).map(decode);
  }
  postMessage({
    type: "result", text, causal, layers: W.layers,
    tokens: ids.map(decode), attn: out.attn, simple: ids.length ? simplified(W, ids) : [],
    next: out.next && out.next.map(decode), nextMasked, truncated: tk.encode(text).length > MAX_TOKENS,
  });
  running = false;
  if (again) { again = false; answer(); }
}

const progress = () => postMessage({ type: "progress", layers: W.layers, shards: W.wte.filter(Boolean).length, allShards: W.allShards, mb: bytes / 1e6 });

async function boot() {
  const [manifest, enc, bpe] = await Promise.all([
    fetch("weights/manifest.json").then(r => r.json()), fetch("gpt2/encoder.json").then(r => r.json()), fetch("gpt2/vocab.bpe").then(r => r.text()),
  ]);
  tk = new GPT2(enc, bpe);
  W = new Weights(manifest, async f => { const b = await fetchBuf(f); bytes += b.byteLength; progress(); return b; });
  await W.loadBase();
  answer();
  for (let i = 0; i < 12; i++) { await W.loadLayer(i); progress(); answer(); }
  for (let k = 0; k < W.allShards; k++) await W.loadShard(k);  // the output layer needs the whole vocabulary
  progress();
  answer();
}

onmessage = e => { latest = e.data; answer(); };
boot().catch(err => postMessage({ type: "error", message: String(err) }));
