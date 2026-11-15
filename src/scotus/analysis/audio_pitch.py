"""
The audio trial, step 1: the recordings of the arguments and the pitch of every turn.

    python -I -m scotus.analysis.audio_pitch --list                 which recordings, and how large
    python -I -m scotus.analysis.audio_pitch --download             -> data/audio/mp3/<id>.mp3
    python -I -m scotus.analysis.audio_pitch --pitch                -> data/audio/turn_pitch.parquet

Source. Oyez's transcript JSON (scrape_oyez.py --media-ids, data/oyez/raw/media_<id>.json)
names the mp3 of each argument in Oyez's storage (s3.amazonaws.com/oyez.case-media.mp3,
CC BY-NC 4.0) and gives the start and stop of every turn in that file. Our transcript ids
are Oyez's audio ids, and turn_idx counts the turns of the JSON in (section, turn) order,
as in prepare_data.py, so a row here joins turns.parquet on (convo_id, turn_idx).

Pitch. The mp3 is decoded by ffmpeg to mono 16 kHz. Praat (praat-parselmouth) tracks the
fundamental frequency every 10 ms (autocorrelation, floor 60 Hz, ceiling 400 Hz) and the
intensity. Per turn, over its voiced frames (the first and last 50 ms left out):
    n_voiced, f0_med (Hz), st_med and st_sd (semitones re 100 Hz), int_db (mean dB).
The settings were fixed before any recording was processed.

Rules followed: one file at a time with a pause; a generic User-Agent; only addresses in
Oyez's mp3 bucket are fetched (the address comes from a downloaded file); a file is kept
only when as many bytes arrived as the server announced, and files on disk are skipped, so
an interrupted run resumes. Run with -I (isolated
mode) because the script reads downloaded files; it imports no project module.
"""
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

AGENT = "cs230-course-project-script/1.0 (academic, non-commercial)"
BUCKET = ("s3.amazonaws.com", "/oyez.case-media.mp3/")
SR, STEP, FLOOR, CEIL, EDGE, MIN_VOICED = 16000, 0.01, 60.0, 400.0, 0.05, 20


def listing(ids, raw):
    """[(id, href, size)] for the recordings that have an mp3; ids without one."""
    rows, missing = [], []
    for i in ids:
        f = raw / f"media_{i}.json"
        if not f.exists():
            missing.append((i, "transcript JSON not on disk"))
            continue
        mp3 = [m for m in (json.loads(f.read_text()).get("media_file") or [])
               if m and m.get("mime") == "audio/mpeg" and m.get("href")]
        if not mp3:
            missing.append((i, "no mp3 listed"))
            continue
        u = urllib.parse.urlparse(mp3[0]["href"])
        if u.scheme != "https" or u.netloc != BUCKET[0] or not u.path.startswith(BUCKET[1]):
            missing.append((i, f"address outside Oyez's mp3 bucket: {mp3[0]['href'][:60]}"))
            continue
        rows.append((i, mp3[0]["href"], int(mp3[0].get("size") or 0)))
    return rows, missing


def fetch_file(url, path, pause):
    """GET url -> path (atomic). True if the whole file is on disk.

    Complete = as many bytes as the server announced. The size in Oyez's JSON is not used
    for this: for about one recording in eight it is off by a fraction of a percent."""
    if path.exists() and path.stat().st_size > 0:
        return True
    errors = offline = 0
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT})
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                size = int(r.headers.get("Content-Length") or 0)
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            got = tmp.stat().st_size
            if size and got != size:
                raise ValueError(f"{got} bytes received, {size} announced")
            tmp.rename(path)
            time.sleep(pause)
            return True
        except (urllib.error.HTTPError, ValueError) as e:       # the server answered badly
            errors += 1
            if errors >= 3:
                print(f"    FAILED {url}: {e}", flush=True)
                time.sleep(pause)
                return False
            time.sleep(pause * (errors + 1))
        except Exception as e:                                  # no connection: wait it out
            offline += 1
            if offline in (1, 10) or offline % 60 == 0:
                print(f"    no connection ({type(e).__name__}); waiting, try {offline}", flush=True)
            time.sleep(60)


def turn_times(media):
    """(start, stop) of every turn, in the order that defines turn_idx."""
    out = []
    for sec in (media.get("transcript") or {}).get("sections") or []:
        for t in sec.get("turns") or []:
            out.append((float(t.get("start") or 0), float(t.get("stop") or 0)))
    return out


def pitch_one(job):
    """Per-turn pitch table of one recording -> a parquet part. Returns a log line."""
    import numpy as np
    import pandas as pd
    import parselmouth
    i, mp3, raw, part = job
    t0 = time.time()
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(mp3), "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(pcm, dtype=np.float32).astype(np.float64)
    snd = parselmouth.Sound(x, sampling_frequency=SR)
    pitch = snd.to_pitch_ac(time_step=STEP, pitch_floor=FLOOR, pitch_ceiling=CEIL)
    f0, tf = pitch.selected_array["frequency"], pitch.xs()
    inten = snd.to_intensity(minimum_pitch=FLOOR, time_step=STEP)
    db = np.interp(tf, inten.xs(), inten.values[0])
    st = 12 * np.log2(np.where(f0 > 0, f0, np.nan) / 100.0)
    rows = []
    for k, (a, b) in enumerate(turn_times(json.loads((raw / f"media_{i}.json").read_text()))):
        lo, hi = np.searchsorted(tf, [a + EDGE, b - EDGE])
        v = f0[lo:hi] > 0
        n = int(v.sum())
        ok = n >= MIN_VOICED
        rows.append((str(i), k, a, b, int(max(hi - lo, 0)), n,
                     float(np.median(f0[lo:hi][v])) if ok else np.nan,
                     float(np.nanmedian(st[lo:hi])) if ok else np.nan,
                     float(np.nanstd(st[lo:hi])) if ok else np.nan,
                     float(db[lo:hi][v].mean()) if ok else np.nan))
    df = pd.DataFrame(rows, columns=["convo_id", "turn_idx", "start", "stop", "n_frames", "n_voiced",
                                     "f0_med", "st_med", "st_sd", "int_db"])
    df["audio_s"] = len(x) / SR
    df.to_parquet(part, index=False)
    last = max((b for _, b in zip(range(len(rows)), df.stop)), default=0.0)
    return (f"{i}: {len(x) / SR / 60:.0f} min of audio, last turn ends at {last / 60:.0f} min, "
            f"{len(df)} turns, {int(df.st_med.notna().sum())} with pitch, {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="data/audio/trial_ids.txt")
    ap.add_argument("--raw", default="data/oyez/raw")
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--pitch", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N recordings (to try things out)")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between downloads")
    ap.add_argument("--max-gb", type=float, default=10.0, help="refuse to download more than this")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    ids = [x for x in Path(args.ids).read_text().split() if x.strip()]
    rows, missing = listing(ids, raw)
    if args.limit:
        rows = rows[:args.limit]
    total = sum(s for _, _, s in rows)
    print(f"{len(ids)} transcripts listed; {len(rows)} recordings with an mp3, {total / 1e9:.2f} GB "
          f"(mean {total / max(len(rows), 1) / 1e6:.1f} MB); {len(missing)} without", flush=True)
    for i, why in missing[:10]:
        print(f"  {i}: {why}", flush=True)

    if args.download:
        if total / 1e9 > args.max_gb:
            raise SystemExit(f"{total / 1e9:.2f} GB is more than --max-gb {args.max_gb}: nothing downloaded.")
        (out / "mp3").mkdir(parents=True, exist_ok=True)
        done = got = 0
        for k, (i, href, size) in enumerate(rows):
            ok = fetch_file(href, out / "mp3" / f"{i}.mp3", args.pause)
            done += ok
            got += (out / "mp3" / f"{i}.mp3").stat().st_size if ok else 0
            if (k + 1) % 10 == 0 or k + 1 == len(rows):
                print(f"  {k + 1}/{len(rows)} recordings, {got / 1e9:.2f} GB on disk, "
                      f"{k + 1 - done} failed", flush=True)
        print("download finished", flush=True)

    if args.pitch:
        import multiprocessing as mp
        import pandas as pd
        parts = out / "pitch_parts"
        parts.mkdir(parents=True, exist_ok=True)
        have = [(i, out / "mp3" / f"{i}.mp3") for i, _, _ in rows if (out / "mp3" / f"{i}.mp3").exists()]
        jobs = [(i, p, raw, parts / f"{i}.parquet") for i, p in have if not (parts / f"{i}.parquet").exists()]
        print(f"{len(have)} recordings on disk, {len(jobs)} to process with {args.workers} workers", flush=True)
        with mp.Pool(args.workers) as pool:
            for k, line in enumerate(pool.imap_unordered(pitch_one, jobs)):
                if (k + 1) % 10 == 0 or k + 1 == len(jobs) or k < 3:
                    print(f"  {k + 1}/{len(jobs)}  {line}", flush=True)
        df = pd.concat([pd.read_parquet(parts / f"{i}.parquet") for i, _ in have], ignore_index=True)
        df.to_parquet(out / "turn_pitch.parquet", index=False)
        print(f"wrote {out / 'turn_pitch.parquet'}: {len(df):,} turns of {df.convo_id.nunique()} "
              f"recordings, pitch on {df.st_med.notna().mean():.1%}", flush=True)


if __name__ == "__main__":
    main()
