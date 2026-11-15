"""
Flat-transcript baseline, step 1: one frozen ModernBERT vector per case. This is the
baseline the proposal calls "flat ModernBERT on the first 8k tokens": the transcript as
one long text with no turn structure.

    python -I -m scotus.features.flat_encode --limit 6        # timing test on a few cases, nothing saved
    caffeinate -i python -I -m scotus.features.flat_encode    # all cases from 1980 -> embeddings/flat_mb_*.npy
    python -I -m scotus.features.flat_encode --data data/ext --out embeddings/ext      # the 2020-25 terms
    python -I -m scotus.features.flat_encode --rest           # afterwards: the tokens beyond the first window
                                              # -> flat_mb_meanall.npy (whole transcript)
(-I: isolated mode, because it reads downloaded model files.)

Model: answerdotai/ModernBERT-base (149M parameters, 22 layers, reads up to 8,192 tokens;
Apache-2.0), pinned to the revision below, safetensors weights.

The text. One line per turn, in argument order, the first 8,192 tokens kept:
    Roberts: We'll hear argument this morning in Case 17-1299 ...
    Petitioner: Mr. Chief Justice, and may it please the Court: ...
    Kagan: Mr. Smith, what about ...
A justice is labelled by surname; an advocate by the side they argue for (Petitioner,
Respondent, Amicus, Counsel if the side is unknown); a turn without a speaker record is
"Unknown". That is what a reader of the printed transcript knows, in as few tokens as
possible. Nothing else is given: no counts, no positions, no list of who votes.

Outputs (in embeddings/, gitignored; row i = row i of the index):
    flat_mb_mean.npy     [n_cases, 768]  mean over the real tokens of the last layer
    flat_mb_cls.npy      [n_cases, 768]  the first ([CLS]) token of the last layer
    flat_mb_clsall.npy   [n_cases, 768]  (--rest) the windows' [CLS] vectors, averaged by length
    flat_mb_meanall.npy  [n_cases, 768]  (--rest) mean over the tokens of the WHOLE transcript:
                           the text after the first window goes through the model in
                           further windows of 8,192 tokens, and the windows' means are
                           averaged by their length. Windows do not see each other. Added
                           because the first window holds 66% of a typical transcript up to
                           2019 but only 44% from 2020 on.
    flat_mb_index.parquet  case_id, year, n_tokens (used), n_tokens_full, n_turns,
                           n_turns_seen, resp_seen (does the respondent's advocate start
                           inside the window?)
Only the transcripts are read. No vote or outcome column is opened.
"""

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MODEL = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
MAX_TOKENS = 8192
SIDE = {"petitioner": "Petitioner", "respondent": "Respondent", "amicus": "Amicus",
        "advocate_unknown": "Counsel"}


def speaker_label(speaker_id, role):
    if role == "justice":                       # j__john_g_roberts_jr -> Roberts
        return re.sub(r"^j__|_jr$|\d+$", "", speaker_id).split("_")[-1].capitalize()
    return SIDE.get(role, "Unknown")


def flat_texts(turns):
    """case_id -> (text, [character offset of each line], [role of each line]).
    `turns`: case_id, speaker_id, role, text, pos; any order."""
    out = {}
    t = turns.sort_values(["case_id", "pos"])
    for case_id, g in t.groupby("case_id", sort=False):
        lines, starts, at = [], [], 0
        for sid, role, text in zip(g.speaker_id, g.role, g.text):
            line = f"{speaker_label(sid, role)}: {' '.join(str(text).split())}"
            lines.append(line); starts.append(at)
            at += len(line) + 1
        out[case_id] = ("\n".join(lines), np.array(starts), g.role.to_numpy())
    return out


def tokenize(texts, tok):
    """Per case: the first MAX_TOKENS token ids, and how much of the transcript they cover."""
    rows = []
    for case_id, (text, starts, roles) in texts.items():
        enc = tok(text, add_special_tokens=True, return_offsets_mapping=True)
        ids, n_full = enc["input_ids"], len(enc["input_ids"])
        cut = len(text)
        if n_full > MAX_TOKENS:
            cut = enc["offset_mapping"][MAX_TOKENS - 2][1]      # end of the last token kept
            ids = ids[:MAX_TOKENS - 1] + [tok.sep_token_id]
        seen = starts < cut
        rows.append({"case_id": case_id, "ids": np.asarray(ids, dtype=np.int64),
                     "ids_full": np.asarray(enc["input_ids"], dtype=np.int64),
                     "n_tokens": len(ids), "n_tokens_full": n_full, "n_turns": len(starts),
                     "n_turns_seen": int(seen.sum()),
                     "resp_seen": bool((seen & (roles == "respondent")).any())})
    return pd.DataFrame(rows)


def rest_windows(ids_full, cls_id, sep_id):
    """The tokens after the first window, as further windows of at most MAX_TOKENS, each
    with its own [CLS] and [SEP]."""
    body = ids_full[MAX_TOKENS - 1:-1]              # what the first window left out
    step = MAX_TOKENS - 2
    return [np.concatenate([[cls_id], body[a:a + step], [sep_id]])
            for a in range(0, len(body), step)]


def token_batches(lengths, budget):
    """Indices grouped longest first, so that width x batch size stays within `budget`."""
    batches, cur, width = [], [], 0
    for i in np.argsort(-np.asarray(lengths), kind="stable"):
        if cur and max(width, lengths[i]) * (len(cur) + 1) > budget:
            batches.append(cur); cur, width = [], 0
        cur.append(int(i)); width = max(width, lengths[i])
    return batches + ([cur] if cur else [])


@torch.no_grad()
def encode(model, ids_list, pad_id, device):
    """(mean-pooled, [CLS]) vectors of the last layer for a batch of id arrays."""
    width = max(len(i) for i in ids_list)
    ids = torch.full((len(ids_list), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids_list), width), dtype=torch.long)
    for b, i in enumerate(ids_list):
        ids[b, :len(i)] = torch.from_numpy(i)
        mask[b, :len(i)] = 1
    h = model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state.float()
    m = mask.to(device).unsqueeze(-1).float()
    mean = (h * m).sum(1) / m.sum(1)
    return mean.cpu().numpy(), h[:, 0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="embeddings")
    ap.add_argument("--start", type=int, default=1980, help="first term to encode (the "
                    "networks train from 1980; earlier transcripts are not needed)")
    ap.add_argument("--limit", type=int, help="timing test on this many cases; saves nothing")
    ap.add_argument("--rest", action="store_true",
                    help="encode the tokens beyond the first window too (after a normal run)")
    ap.add_argument("--budget", type=int, default=MAX_TOKENS,
                    help="tokens per batch (8,192 = one full-length case at a time)")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32

    cases = pd.read_parquet(data / "cases.parquet", columns=["case_id", "year"])
    cases = cases[cases.year >= args.start]
    if args.limit:
        cases = cases.sample(args.limit, random_state=0)
    turns = pd.read_parquet(data / "turns.parquet",
                            columns=["case_id", "speaker_id", "role", "text"])
    turns["pos"] = pd.read_parquet(data / "turn_feats.parquet", columns=["pos"]).pos.to_numpy()
    turns = turns[turns.case_id.isin(set(cases.case_id))]

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    index = tokenize(flat_texts(turns), tok).merge(cases, on="case_id")
    index = index.sort_values(["year", "case_id"]).reset_index(drop=True)
    cut = index.n_tokens_full > MAX_TOKENS
    print(f"{len(index):,} cases tokenized in {time.time() - t0:.0f}s; longer than "
          f"{MAX_TOKENS:,} tokens: {100 * cut.mean():.1f}%; median share of the transcript "
          f"kept: {100 * (index.n_tokens / index.n_tokens_full).median():.0f}%; respondent's "
          f"advocate inside the window: {100 * index.resp_seen.mean():.1f}%", flush=True)

    model = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=dtype).to(device).eval()
    if args.rest:
        saved = pd.read_parquet(out / "flat_mb_index.parquet")
        assert saved.case_id.tolist() == index.case_id.tolist() and \
            (saved.n_tokens_full.to_numpy() == index.n_tokens_full.to_numpy()).all(), \
            "the saved index does not match these transcripts"
        n_first = index.n_tokens.to_numpy().astype(np.float64)
        total = np.load(out / "flat_mb_mean.npy").astype(np.float64) * n_first[:, None]
        total_cls = np.load(out / "flat_mb_cls.npy").astype(np.float64) * n_first[:, None]
        count = n_first.copy()
        wins = [(i, w) for i in range(len(index))
                for w in rest_windows(index.ids_full[i], tok.cls_token_id, tok.sep_token_id)]
        batches = token_batches([len(w) for _, w in wins], args.budget)
        print(f"{len(wins):,} further windows ({sum(len(w) for _, w in wins) / 1e6:.1f}M tokens) "
              f"for {(index.n_tokens_full > MAX_TOKENS).sum():,} cases", flush=True)
        t0 = time.time()
        for b, rows in enumerate(batches):
            m, c = encode(model, [wins[r][1] for r in rows], tok.pad_token_id, device)
            for r, v, v_cls in zip(rows, m, c):
                i, w = wins[r]
                total[i] += v.astype(np.float64) * len(w)
                total_cls[i] += v_cls.astype(np.float64) * len(w)
                count[i] += len(w)
            if (b + 1) % 200 == 0 or b + 1 == len(batches):
                el = time.time() - t0
                print(f"  {b + 1:,}/{len(batches):,} batches  {el / 60:.1f} min elapsed, about "
                      f"{el / (b + 1) * (len(batches) - b - 1) / 60:.0f} min left", flush=True)
        whole = (total / count[:, None]).astype(np.float32)
        n_full = index.n_tokens_full.to_numpy()
        n_more = np.ceil(np.maximum(n_full - MAX_TOKENS, 0) / (MAX_TOKENS - 2))
        assert np.isfinite(whole).all() and (count == n_full + 2 * n_more).all(), \
            "window bookkeeping is off"                # every token once, plus [CLS] and [SEP]
        np.save(out / "flat_mb_meanall.npy", whole)
        np.save(out / "flat_mb_clsall.npy", (total_cls / count[:, None]).astype(np.float32))
        print(f"wrote {out}/flat_mb_meanall.npy and flat_mb_clsall.npy ({len(index):,} cases, "
              f"{(time.time() - t0) / 60:.1f} min)")
        return
    mean = np.zeros((len(index), model.config.hidden_size), dtype=np.float32)
    cls = np.zeros_like(mean)
    done = np.zeros(len(index), dtype=bool)
    partial = out / "flat_mb_partial.npz"
    if partial.exists() and not args.limit:                    # resume an interrupted run
        p = np.load(partial, allow_pickle=False)
        if list(p["case_id"]) == index.case_id.tolist():
            mean, cls, done = p["mean"], p["cls"], p["done"]
            print(f"resuming: {int(done.sum()):,} cases already encoded", flush=True)

    order = [i for i in np.argsort(-index.n_tokens.to_numpy(), kind="stable") if not done[i]]
    batches, cur, width = [], [], 0
    for i in order:                                            # longest first, by token budget
        n = index.n_tokens[i]
        if cur and max(width, n) * (len(cur) + 1) > args.budget:
            batches.append(cur); cur, width = [], 0
        cur.append(i); width = max(width, n)
    if cur:
        batches.append(cur)

    t0, n_tok = time.time(), 0
    for b, rows in enumerate(batches):
        m, c = encode(model, [index.ids[i] for i in rows], tok.pad_token_id, device)
        mean[rows], cls[rows], done[rows] = m, c, True
        n_tok += int(index.n_tokens[rows].sum())
        if (b + 1) % 100 == 0 or b + 1 == len(batches):
            el = time.time() - t0
            print(f"  {int(done.sum()):,}/{len(index):,} cases  {n_tok / el:,.0f} tokens/s  "
                  f"{el / 60:.1f} min elapsed, about {el / (b + 1) * (len(batches) - b - 1) / 60:.0f} "
                  f"min left", flush=True)
            if not args.limit:
                np.savez(partial, mean=mean, cls=cls, done=done,
                         case_id=index.case_id.to_numpy().astype(str))
    assert np.isfinite(mean).all() and np.isfinite(cls).all(), "non-finite vectors"
    if args.limit:
        print(f"timing test: {len(index)} cases in {time.time() - t0:.1f}s; nothing saved")
        return
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "flat_mb_mean.npy", mean)
    np.save(out / "flat_mb_cls.npy", cls)
    index.drop(columns=["ids", "ids_full"]).assign(row=np.arange(len(index))).to_parquet(
        out / "flat_mb_index.parquet", index=False)
    partial.unlink(missing_ok=True)
    print(f"wrote {out}/flat_mb_mean.npy, flat_mb_cls.npy, flat_mb_index.parquet "
          f"({len(index):,} cases, {n_tok / 1e6:.1f}M tokens, {(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
