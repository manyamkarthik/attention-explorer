// Does the browser model match Python? node scripts/check_js.js (after scripts/export_weights.py)
const fs = require("fs"), path = require("path");
const site = path.join(__dirname, "..", "site");
const { GPT2, Weights, forward, simplified } = require(path.join(site, "model.js"));
const read = f => { const b = fs.readFileSync(path.join(site, f)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.length); };

(async () => {
  const ref = JSON.parse(fs.readFileSync(path.join(site, "weights/reference.json")));
  const tk = new GPT2(JSON.parse(fs.readFileSync(path.join(site, "gpt2/encoder.json"))), fs.readFileSync(path.join(site, "gpt2/vocab.bpe"), "utf8"));
  const W = new Weights(JSON.parse(fs.readFileSync(path.join(site, "weights/manifest.json"))), async f => read(f));
  await W.loadBase();
  for (let k = 0; k < W.allShards; k++) await W.loadShard(k);
  for (let i = 0; i < 12; i++) await W.loadLayer(i);
  const ids = tk.encode(ref.text), t0 = Date.now(), out = forward(W, ids);
  const ms = Date.now() - t0, T = ids.length;
  let diff = 0;
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) diff = Math.max(diff, Math.abs(out.attn[4][3][i * T + j] - ref.attn_4_3[i][j]));
  const guesses = out.next.map(i => show(tk, i));
  console.log("tokens   :", JSON.stringify(tk.tokens(ref.text)) === JSON.stringify(ref.tokens) ? "match" : "DIFFER");
  console.log(`attn 4.3 : max diff vs python ${diff.toExponential(2)}   it->animal ${out.attn[4][3][8 * T + 1].toFixed(3)}`);
  console.log("next     :", guesses.every((g, i) => g === ref.next[i]) ? "match" : "DIFFER", JSON.stringify(guesses));
  console.log(`forward  : ${ms} ms for ${T} tokens`);
  const peek = forward(W, ids, false);
  console.log("no mask  :", JSON.stringify(peek.next.map(i => show(tk, i))));
  const s = simplified(W, ids);
  console.log("simplified row 'it':", Array.from(s.slice(8 * T, 9 * T), v => v.toFixed(2)).join(" "));
})();

function show(tk, id) { return new TextDecoder().decode(Uint8Array.from([...tk.id2tok[id]], c => tk.u2b[c])); }
