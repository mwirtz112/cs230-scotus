"""
Download oral-argument transcripts from the Oyez API for terms the ConvoKit corpus does
not cover (terms 2020 on).

    python -I -m scotus.data.scrape_oyez --terms 2020 2021 2022 2023 2024 2025
    python -I -m scotus.data.scrape_oyez --terms 2017 --limit 30      # a sample of a term we already have,
                                                          # to check the parser against ConvoKit
    python -I -m scotus.data.scrape_oyez --media-ids ids.txt          # only the listed transcripts (their
                                                          # JSON names the audio files)

Source: https://api.oyez.org (the Oyez Project; the ConvoKit Supreme Court corpus was
built from the same transcripts). Three kinds of request, each saved unchanged as JSON:
    cases?per_page=0&filter=term:T          -> data/oyez/raw/term_T.json   (case list)
    cases/T/<docket>                        -> data/oyez/raw/case_T_<docket>.json
    case_media/oral_argument_audio/<id>     -> data/oyez/raw/media_<id>.json (the transcript)

Rules followed:
  * robots.txt of api.oyez.org sets "Crawl-delay: 10": one request every 10 seconds
    (--delay). About 130 requests per term, so a term takes roughly 20 minutes.
  * Only cases with an "Argued" event are fetched. Files already on disk are skipped, so
    an interrupted run resumes.
  * A generic User-Agent; nothing about the user is sent.
  * Standard library only, and run with -I (isolated mode), because the script reads the
    JSON it has just downloaded to find the transcript links.

The case files also contain how the case was decided. Nothing in this script reads those
fields, and prepare_oyez.py does not either: votes come from the SCDB join, under the same
test lock as the rest of the project.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.oyez.org"
AGENT = "cs230-course-project-script/1.0 (academic, non-commercial)"


def fetch(url, path, delay, retries=4):
    """GET url -> path (atomic). Returns True if a request was made.

    If the server answers with an error, give up after a few tries. If there is no
    connection at all (wifi down, laptop asleep), wait a minute and try again for as long
    as it takes, so an outage only pauses the download."""
    if path.exists() and path.stat().st_size > 2:
        return False
    errors = offline = 0
    while True:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": AGENT,
                                                       "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            json.loads(body)                                # must be valid JSON
            tmp = path.with_suffix(".part")
            tmp.write_bytes(body)
            tmp.rename(path)
            time.sleep(delay)
            return True
        except (urllib.error.HTTPError, ValueError) as e:   # the server answered badly
            errors += 1
            if errors >= retries:
                print(f"    FAILED {url}: {e}", flush=True)
                time.sleep(delay)
                return True
            print(f"    retry {errors} for {url}: {e}", flush=True)
            time.sleep(delay * (errors + 1))
        except Exception as e:                              # no connection: wait it out
            offline += 1
            if offline in (1, 10) or offline % 60 == 0:
                print(f"    no connection ({type(e).__name__}); waiting, try {offline}",
                      flush=True)
            time.sleep(60)


def safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terms", nargs="+", type=int, default=[])
    ap.add_argument("--media-ids", help="a text file of Oyez argument-audio ids, one per line: "
                    "fetch only those transcripts (media_<id>.json), e.g. for terms whose "
                    "cases we already have from ConvoKit")
    ap.add_argument("--out", default="data/oyez/raw")
    ap.add_argument("--delay", type=float, default=10.0, help="seconds between requests")
    ap.add_argument("--limit", type=int, help="only the first N argued cases of each term")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.terms and not args.media_ids:
        ap.error("give --terms or --media-ids")

    if args.media_ids:
        ids = [x.strip() for x in Path(args.media_ids).read_text().split() if x.strip()]
        todo = [i for i in ids if not (out / f"media_{safe(i)}.json").exists()]
        print(f"{len(ids)} transcripts listed, {len(ids) - len(todo)} on disk, {len(todo)} to fetch "
              f"(about {len(todo) * args.delay / 60:.0f} minutes)", flush=True)
        for k, i in enumerate(todo):
            fetch(f"{API}/case_media/oral_argument_audio/{i}", out / f"media_{safe(i)}.json", args.delay)
            if (k + 1) % 10 == 0 or k + 1 == len(todo):
                print(f"  {k + 1}/{len(todo)} transcripts", flush=True)

    for term in args.terms:
        listing = out / f"term_{term}.json"
        fetch(f"{API}/cases?per_page=0&filter=term:{term}", listing, args.delay)
        if not listing.exists():
            continue
        cases = json.loads(listing.read_text())
        argued = [c for c in cases
                  if any(t and t.get("event") in ("Argued", "Reargued")
                         for t in (c.get("timeline") or []))]
        if args.limit:
            argued = argued[:args.limit]
        print(f"term {term}: {len(cases)} cases listed, {len(argued)} argued to fetch", flush=True)
        n_media = 0
        for i, c in enumerate(argued):
            docket = c.get("docket_number")
            case_file = out / f"case_{term}_{safe(docket)}.json"
            fetch(c["href"], case_file, args.delay)
            if not case_file.exists():
                continue
            case = json.loads(case_file.read_text())
            for audio in case.get("oral_argument_audio") or []:
                if not audio or audio.get("unavailable") or not audio.get("href"):
                    continue
                fetch(audio["href"], out / f"media_{safe(audio['id'])}.json", args.delay)
                n_media += 1
            if (i + 1) % 10 == 0 or i + 1 == len(argued):
                print(f"  term {term}: {i + 1}/{len(argued)} cases, {n_media} transcripts",
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
