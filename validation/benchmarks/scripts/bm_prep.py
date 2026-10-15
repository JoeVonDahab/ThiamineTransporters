"""Build the combined benchmark variant table (CV set + ClinVar time-split) with
normalised GRCh38 VCF-style alleles, TripleVar scores, labels and trusted flags."""
import re
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/joe/projects/TripleVar/src')
from triplevar.metrics import Scores  # noqa: E402

R = '/home/joe/projects/TripleVar/'
OUT = sys.argv[1]


def read_fa(p):
    lines = open(p).read().splitlines()
    m = re.match(r'>(\S+):(\d+)-(\d+)', lines[0])
    start, end = int(m.group(2)), int(m.group(3))
    seq = ''.join(lines[1:]).upper()
    assert len(seq) == end - start + 1, (p, len(seq), end - start + 1)
    return start, seq


FA = {'1': read_fa(R + 'data/reference/SLC19A2.window.fa'), '2': read_fa(R + 'data/reference/SLC19A3.window.fa')}


def refseq(chrom, pos, length):
    start, seq = FA[chrom]
    i = pos - start
    assert i >= 0 and i + length <= len(seq)
    return seq[i:i + length]


def normalize(chrom, p, r, a):
    """vt-style normalisation: right-trim, left-extend when an allele empties, then left-trim to 1 anchor base."""
    while True:
        changed = False
        if len(r) > 0 and len(a) > 0 and r[-1] == a[-1]:
            r, a = r[:-1], a[:-1]
            changed = True
        if len(r) == 0 or len(a) == 0:
            b = refseq(chrom, p - 1, 1)
            r, a, p = b + r, b + a, p - 1
            changed = True
        if not changed:
            break
    while len(r) > 1 and len(a) > 1 and r[0] == a[0]:
        r, a, p = r[1:], a[1:], p + 1
    return p, r, a


def dbsnp_to_vcf(chrom, pos, orig, mut):
    ref = '' if orig == '-' else orig
    alt = '' if mut == '-' else mut
    if ref == '':  # insertion after base `pos` (matches the time-split matcher convention)
        anchor = refseq(chrom, pos, 1)
        p, r, a = pos, anchor, anchor + alt
    else:
        ok = refseq(chrom, pos, len(ref)) == ref
        if not ok:
            return None
        if len(ref) == 1 and len(alt) == 1:
            return pos, ref, alt
        anchor = refseq(chrom, pos - 1, 1)
        p, r, a = pos - 1, anchor + ref, anchor + alt
    return normalize(chrom, p, r, a)


def vtype(r, a):
    if len(r) == 1 and len(a) == 1:
        return 'SNV'
    if len(r) == len(a):
        return 'MNV'
    return 'indel'


rows = []
# ---------------- A) cross-validation set ----------------
lab = pd.read_csv(R + 'data/processed/labeled.csv')
oof = pd.read_csv(R + 'outputs/final/production.oof.csv')
nz = lab[lab.y != 0].reset_index()
assert len(nz) == len(oof) and (nz.snp_id.values == oof.snp_id.values).all() and (nz.y.values == oof.y.values).all()
probs = oof[['p_P', 'p_LP', 'p_LB', 'p_B', 'p_normal']].values
sc = Scores(probs)
assert list(sc.classes) == [-2, -1, 1, 2, 3], sc.classes
nz['triplevar_score'] = sc.pathogenic_given_variant
nz['triplevar_call'] = sc.call_grouped()
nz['triplevar_argmax'] = sc.pred
cv = nz[nz.y.isin([-2, -1, 1, 2])].copy()
assert len(cv) == 864

au = pd.read_csv(R + 'validation/external/training_label_audit.csv')
key = ['snp_id', 'pos', 'original', 'mutant']
assert not au.duplicated(key).any()
cv = cv.merge(au[key + ['label_source', 'training_label']], on=key, how='left', validate='one_to_one')
print('CV audit unmatched:', cv.label_source.isna().sum())
cv['trusted'] = cv.label_source.eq('own_allele')

bad = []
for _, x in cv.iterrows():
    chrom = str(int(x['#chr']))
    v = dbsnp_to_vcf(chrom, int(x.pos), x.original, x.mutant)
    if v is None:
        bad.append((x.snp_id, x.pos, x.original, x.mutant))
        p, r, a = int(x.pos), x.original, x.mutant
    else:
        p, r, a = v
    rows.append(dict(
        set='cv', variant_key=f'rs{x.snp_id}|{chrom}:{p}:{r}:{a}', snp_id=x.snp_id, gene=x.gene, chrom=chrom, pos=p,
        ref=r, alt=a, orig_repr=f'{x.pos}:{x.original}>{x.mutant}', vtype=vtype(r, a),
        consequence=x.function_class, y=int(x.y), label_binary=int(x.y in (-2, -1)),
        label_source=x.label_source, trusted=bool(x.trusted), triplevar_score=float(x.triplevar_score),
        triplevar_call=bool(x.triplevar_call), triplevar_abstract_score=np.nan, ref_ok=v is not None))
print('CV ref mismatches vs reference window:', len(bad), bad[:10])

# ---------------- B) ClinVar time-split ----------------
ts = pd.read_csv(R + 'reports/predictions/clinvar_timesplit.triplevar_v1.csv')
ab = pd.read_csv('/tmp/ts_abstract.csv')
assert len(ab) == len(ts) and (ab.variant_id.values == ts.variant_id.values).all()
ts['abstract'] = ab.p_pathogenic.values
ts['abstract_call'] = ab.pathogenic_call.values
assert ts[['chrom', 'pos', 'ref', 'alt']].notna().all().all()
for _, x in ts.iterrows():
    chrom = str(int(x.chrom))
    pos0, r0, a0 = int(x.pos), x.ref, x.alt
    ok = refseq(chrom, pos0, len(r0)) == r0
    p, r, a = normalize(chrom, pos0, r0, a0) if (len(r0) > 1 or len(a0) > 1) else (pos0, r0, a0)
    rows.append(dict(
        set='timesplit', variant_key=x.variant_id, snp_id=np.nan, gene=x.gene, chrom=chrom, pos=p, ref=r, alt=a,
        orig_repr=f'{pos0}:{r0}>{a0}', vtype=vtype(r, a), consequence=x.consequence,
        is_missense_source=bool(x.is_missense), y=np.nan, label_binary=int(x.label_binary), label_source=np.nan,
        trusted=np.nan, triplevar_score=float(x.p_pathogenic), triplevar_call=bool(x.pathogenic_call),
        triplevar_abstract_score=float(x.abstract), triplevar_abstract_call=bool(x.abstract_call), ref_ok=ok,
        feature_source=x.feature_source))
    if (p, r, a) != (pos0, r0, a0):
        print('TS normalised', x.variant_id, (pos0, r0, a0), '->', (p, r, a))
df = pd.DataFrame(rows)
print(df.groupby(['set', 'vtype']).size())
print('ref_ok false:', df[~df.ref_ok][['set', 'variant_key', 'orig_repr']].to_string())
print(df.groupby('set').label_binary.value_counts())
print('CV trusted:', df[df.set == 'cv'].trusted.value_counts().to_dict())
# overlap between sets
k = df.chrom + ':' + df.pos.astype(str) + ':' + df.ref + ':' + df.alt
df['vcf_key'] = k
both = set(k[df.set == 'cv']) & set(k[df.set == 'timesplit'])
print('variants in both sets:', len(both), sorted(both)[:20])
print('duplicate vcf keys within cv:', df[df.set == 'cv'].vcf_key.duplicated().sum())
d = df[(df.set == 'cv') & df.vcf_key.duplicated(keep=False)].sort_values('vcf_key')
print(d[['variant_key', 'orig_repr', 'y', 'triplevar_score']].to_string())
df.to_csv(OUT, index=False)
print('wrote', OUT, len(df))
