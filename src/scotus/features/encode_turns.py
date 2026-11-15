"""
Encode every turn with a frozen encoder and cache one vector per turn.

Run from the repository root:
    python -m scotus.features.encode_turns --limit 20000     # benchmark on a random sample -> embeddings/bench/
    caffeinate -i python -m scotus.features.encode_turns     # full run (caffeinate keeps the Mac awake)
    python -m scotus.features.encode_turns --data data/ext --out embeddings/ext   # terms 2020-25

Outputs (in embeddings/, gitignored):
    turns_last.npy    [n_turns, 768] fp16  mean over tokens of the last layer   (primary)
    turns_last4.npy   [n_turns, 768] fp16  mean over tokens of the last-4-layer average
    index.parquet     row -> case_id, convo_id, turn_idx, n_tokens, n_chunks
Row i of the arrays is row i of data/turns.parquet.

Encoder comparison. The default is Legal-BERT (nlpaueb/legal-bert-base-uncased),
mean-pooled. Another encoder is selected with --model, pooled the way
its authors define its sentence embedding, and written to its own file:
    python -I -m scotus.features.encode_turns --model BAAI/bge-base-en-v1.5 \
        --revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a --pool cls --normalize \
        --name turns_bge                       # -> embeddings/turns_bge.npy, index_turns_bge.parquet
(-I: isolated mode, because it reads downloaded model files.) With --name only that one array is written (no last-4-layer average).

How it works:
  1. Tokenize every turn (no truncation).
  2. Split long turns into 510-token chunks (+[CLS], [SEP] = 512).
  3. Sort chunks by length and batch them by a token budget, so batches hold
     similar-length chunks and little compute is wasted on padding.
  4. Run the encoder (fp16 on the Apple GPU). Mean-pool the real tokens (not [CLS],
     [SEP] or padding) of the last layer and of the last-4-layer average.
  5. A turn's vector is the token-weighted average of its chunk vectors.
Progress is saved every few hundred batches, so an interrupted run resumes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "nlpaueb/legal-bert-base-uncased"
CHUNK = 510                     # tokens per chunk, leaving room for [CLS] and [SEP]
DIM = 768


def tokenize(texts, tok, bs=20000):
    """Token ids per turn (no special tokens, no truncation)."""
    ids = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], add_special_tokens=False)["input_ids"]
        ids.extend(np.asarray(e, dtype=np.int32) for e in enc)
        print(f"\r  tokenized {min(i + bs, len(texts)):,}/{len(texts):,}", end="", flush=True)
    print()
    return ids


def make_chunks(ids):
    """Split each turn into <= CHUNK-token pieces. Returns chunk ids, owner row, length."""
    chunk_ids, chunk_row = [], []
    for row, arr in enumerate(ids):
        if len(arr) == 0:                       # nothing to encode; gets a zero vector
            chunk_ids.append(arr); chunk_row.append(row)
            continue
        for s in range(0, len(arr), CHUNK):
            chunk_ids.append(arr[s:s + CHUNK]); chunk_row.append(row)
    lens = np.array([len(c) for c in chunk_ids], dtype=np.int32)
    return chunk_ids, np.array(chunk_row, dtype=np.int64), lens


def make_batches(lens, budget):
    """Longest chunks first; each batch has at most `budget` padded tokens."""
    order = np.argsort(-lens, kind="stable")
    batches, cur, cur_max = [], [], 0
    for c in order:
        width = int(lens[c]) + 2                # + [CLS], [SEP]
        if cur and max(cur_max, width) * (len(cur) + 1) > budget:
            batches.append(np.array(cur)); cur, cur_max = [], 0
        cur.append(c); cur_max = max(cur_max, width)
    if cur:
        batches.append(np.array(cur))
    return batches


@torch.inference_mode()
def encode_batch(model, tok, chunk_ids, batch, device, pooling="mean", normalize=False):
    """Pooled last-layer vector (mean over real tokens, or [CLS]) and the mean-pooled
    last-4-layer average for one batch of chunks (fp32)."""
    width = max(len(chunk_ids[c]) for c in batch) + 2
    input_ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
    attn = torch.zeros((len(batch), width), dtype=torch.long)
    pool = torch.zeros((len(batch), width), dtype=torch.float32)
    for k, c in enumerate(batch):
        n = len(chunk_ids[c])
        input_ids[k, 0] = tok.cls_token_id
        input_ids[k, 1:n + 1] = torch.from_numpy(chunk_ids[c].astype(np.int64))
        input_ids[k, n + 1] = tok.sep_token_id
        attn[k, :n + 2] = 1
        pool[k, 1:n + 1] = 1                    # average real tokens only

    out = model(input_ids=input_ids.to(device), attention_mask=attn.to(device),
                output_hidden_states=True)
    hs = out.hidden_states                      # embeddings + 12 layers
    pool = pool.to(device).unsqueeze(-1)
    denom = pool.sum(1).clamp(min=1)
    if pooling == "cls":                        # the [CLS] vector (e.g. BGE models)
        last = hs[-1][:, 0].float()
    else:
        last = (hs[-1].float() * pool).sum(1) / denom
    if normalize:                               # unit length, as sentence encoders define it
        last = torch.nn.functional.normalize(last, dim=-1)
    last4 = (torch.stack(hs[-4:]).float().mean(0) * pool).sum(1) / denom
    return last.cpu().numpy(), last4.cpu().numpy()


def combine(chunk_vecs, chunk_row, lens, n_turns):
    """Turn vector = token-weighted average of its chunk vectors."""
    result = np.zeros((n_turns, DIM), dtype=np.float16)
    n_per_turn = np.bincount(chunk_row, minlength=n_turns)
    single = n_per_turn[chunk_row] == 1                     # ~99% of turns: copy directly
    result[chunk_row[single]] = chunk_vecs[np.flatnonzero(single)].astype(np.float16)
    multi = np.flatnonzero(~single)                         # long turns: weighted average
    multi = multi[np.argsort(chunk_row[multi], kind="stable")]   # group chunks by turn
    rows, starts = np.unique(chunk_row[multi], return_index=True)
    for row, cs in zip(rows, np.split(multi, starts[1:])):
        result[row] = np.average(chunk_vecs[cs], axis=0, weights=lens[cs]).astype(np.float16)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="embeddings")
    ap.add_argument("--limit", type=int, help="benchmark on a random sample of N turns")
    ap.add_argument("--budget", type=int, default=16384, help="padded tokens per batch")
    ap.add_argument("--model", default=MODEL, help="Hugging Face encoder (BERT-style, 768-d)")
    ap.add_argument("--revision", help="commit of the model repository, for reproducibility")
    ap.add_argument("--pool", default="mean", choices=["mean", "cls"],
                    help="how a chunk's token vectors become one vector")
    ap.add_argument("--normalize", action="store_true", help="scale chunk vectors to unit length")
    ap.add_argument("--name", help="output stem, e.g. turns_bge -> embeddings/turns_bge.npy "
                    "(default: turns_last.npy and turns_last4.npy)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    out = Path(args.out) / ("bench" if args.limit else "")
    tmp = out / ("tmp" if not args.name else f"tmp_{args.name}")
    tmp.mkdir(parents=True, exist_ok=True)

    turns = pd.read_parquet(Path(args.data) / "turns.parquet",
                            columns=["case_id", "convo_id", "turn_idx", "text"])
    if args.limit:
        turns = turns.sample(n=args.limit, random_state=0).reset_index(drop=True)
    print(f"{len(turns):,} turns, device={device}, dtype={dtype}")

    # Weights of any other encoder must be in the safetensors format (no pickled code).
    extra = {} if args.model == MODEL else {"use_safetensors": True}
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModel.from_pretrained(args.model, revision=args.revision, dtype=dtype,
                                      **extra).to(device).eval()
    assert model.config.hidden_size == DIM, "this script assumes 768-d encoders"
    print(f"encoder {args.model} (revision {args.revision or 'default'}), pool={args.pool}, "
          f"normalize={args.normalize}")

    t0 = time.time()
    ids = tokenize(turns.text.tolist(), tok)
    chunk_ids, chunk_row, lens = make_chunks(ids)
    batches = make_batches(lens, args.budget)
    print(f"{len(chunk_ids):,} chunks, {int(lens.sum()):,} tokens, {len(batches):,} batches "
          f"(tokenizing took {time.time() - t0:.0f}s)")

    # Chunk vectors go to on-disk arrays; progress.json records finished batches.
    meta = {"n_chunks": len(chunk_ids), "n_batches": len(batches), "budget": args.budget,
            "model": args.model, "pool": args.pool, "normalize": args.normalize}
    prog_path = tmp / "progress.json"
    done = 0
    if prog_path.exists():
        prog = json.loads(prog_path.read_text())
        if {k: prog.get(k) for k in meta} == meta:
            done = prog["done"]
            print(f"Resuming at batch {done:,}/{len(batches):,}")
    mode = "r+" if done else "w+"
    v_last = np.lib.format.open_memmap(tmp / "chunks_last.npy", mode=mode,
                                       dtype=np.float32, shape=(len(chunk_ids), DIM))
    v_last4 = np.lib.format.open_memmap(tmp / "chunks_last4.npy", mode=mode,
                                        dtype=np.float32, shape=(len(chunk_ids), DIM))

    t1, tok_done = time.time(), 0
    for b in range(done, len(batches)):
        batch = batches[b]
        last, last4 = encode_batch(model, tok, chunk_ids, batch, device, args.pool,
                                   args.normalize)
        v_last[batch], v_last4[batch] = last, last4
        tok_done += int(lens[batch].sum())
        if (b + 1) % 200 == 0 or b + 1 == len(batches):
            v_last.flush(); v_last4.flush()
            prog_path.write_text(json.dumps({**meta, "done": b + 1}))
            rate = tok_done / (time.time() - t1)
            left = (int(lens.sum()) - int(lens[np.concatenate(batches[:b + 1])].sum())) / rate
            print(f"\r  batch {b + 1:,}/{len(batches):,}  {rate:,.0f} tokens/s  "
                  f"~{left / 60:.0f} min left", end="", flush=True)
    print()
    enc_time = time.time() - t1

    print("Combining chunks into turns...")
    files = ([("turns_last.npy", v_last), ("turns_last4.npy", v_last4)] if not args.name
             else [(f"{args.name}.npy", v_last)])
    for name, vecs in files:
        np.save(out / name, combine(vecs, chunk_row, lens, len(turns)))
    n_tok = np.array([len(a) for a in ids])
    turns.drop(columns="text").assign(
        row=np.arange(len(turns)), n_tokens=n_tok,
        n_chunks=np.maximum(1, -(-n_tok // CHUNK)),
    ).to_parquet(out / ("index.parquet" if not args.name else f"index_{args.name}.parquet"),
                 index=False)

    # Sanity checks
    E = np.load(out / files[0][0], mmap_mode="r")
    norms = np.linalg.norm(E.astype(np.float32), axis=1)
    print(f"rows {len(E):,} (turns {len(turns):,}); NaN rows {np.isnan(E).any(1).sum()}; "
          f"empty turns {(n_tok == 0).sum()}; norm mean {norms.mean():.2f} "
          f"min {norms.min():.2f} max {norms.max():.2f}")
    print(f"Encoding: {enc_time:.0f}s for {int(lens.sum()):,} tokens "
          f"= {lens.sum() / enc_time:,.0f} tokens/s")
    if args.limit:
        full = pd.read_parquet(Path(args.data) / "turns.parquet", columns=["text"])
        words_all = full.text.str.split().str.len().sum()
        words_s = turns.text.str.split().str.len().sum()
        est = enc_time * words_all / words_s
        print(f"Projected full run: ~{est / 3600:.1f} h encoding "
              f"(+ a few minutes tokenizing and combining)")
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    print(f"Wrote {out}/" + ", ".join(name for name, _ in files))


if __name__ == "__main__":
    main()
