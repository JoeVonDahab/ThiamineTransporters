"""Fetch CADD GRCh38-v1.7 (with annotations) for every benchmark variant, one request per variant,
<=1 request/s, resumable. Raw API responses are appended to a JSONL file."""
import json
import os
import sys
import time
import urllib.request
import urllib.error

import pandas as pd

VARS, OUT = sys.argv[1], sys.argv[2]
BASE = 'https://cadd.gs.washington.edu/api/v1.0/GRCh38-v1.7_inclAnno/'
df = pd.read_csv(VARS, dtype={'chrom': str})
keys = sorted(set(f'{c}:{p}_{r}_{a}' for c, p, r, a in zip(df.chrom, df.pos, df.ref, df.alt)),
              key=lambda s: (len(s.split('_')[1]) + len(s.split('_')[2]) > 2, s))
done = set()
if os.path.exists(OUT):
    for line in open(OUT):
        rec = json.loads(line)
        if rec.get('status') == 200:
            done.add(rec['query'])
todo = [k for k in keys if k not in done]
print(f'{len(keys)} unique variants, {len(done)} cached, {len(todo)} to fetch', flush=True)
last = 0.0
with open(OUT, 'a') as fh:
    for i, k in enumerate(todo):
        for attempt in range(4):
            wait = 1.05 - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            rec = {'query': k, 'url': BASE + k, 'fetched_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            try:
                with urllib.request.urlopen(urllib.request.Request(BASE + k, headers={'User-Agent': 'TripleVar-benchmark/1.0 (academic; polite 1 req/s)'}), timeout=120) as r:
                    rec['status'] = r.status
                    rec['response'] = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                rec['status'] = e.code
                rec['error'] = str(e)
            except Exception as e:  # noqa: BLE001
                rec['status'] = -1
                rec['error'] = repr(e)
            time.sleep(5 * (attempt + 1))
        fh.write(json.dumps(rec) + '\n')
        fh.flush()
        if i % 50 == 0:
            n = len(rec.get('response') or [])
            print(i, k, rec['status'], 'rows', n, time.strftime('%H:%M:%S'), flush=True)
print('done', flush=True)
