"""Head-to-head benchmark: TripleVar vs AlphaMissense, REVEL v1.3 and CADD GRCh38-v1.7.

Inputs : combined variant table from bm_prep.py (argv[1]) + filtered raw score files in
         data/external_raw/benchmarks/.
Outputs: validation/benchmarks/benchmark_scores.csv, benchmark_metrics.csv, figures.
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings('ignore')
R = '/home/joe/projects/TripleVar/'
RAW = R + 'data/external_raw/benchmarks/'
OUTD = R + 'validation/benchmarks/'
N_BOOT = 2000
CANON = {'SLC19A2': 'ENST00000236137', 'SLC19A3': 'ENST00000258403'}  # AM/REVEL transcript for SLC19A3 (MANE = ENST00000644224)

V = pd.read_csv(sys.argv[1], dtype={'chrom': str})
V['k'] = V.chrom + ':' + V.pos.astype(str) + '_' + V.ref + '_' + V.alt

# ----------------------------------------------------------------- AlphaMissense
am = pd.read_csv(RAW + 'AlphaMissense_hg38.SLC19A2_SLC19A3.tsv', sep='\t')
am.columns = [c.lstrip('#') for c in am.columns]
am['k'] = am.CHROM.str.replace('chr', '') + ':' + am.POS.astype(str) + '_' + am.REF + '_' + am.ALT
assert not am.k.duplicated().any()
am = am.set_index('k')
V['alphamissense'] = V.k.map(am.am_pathogenicity)
V['alphamissense_class'] = V.k.map(am.am_class)
V['am_transcript'] = V.k.map(am.transcript_id)
V['am_protein_variant'] = V.k.map(am.protein_variant)

# ----------------------------------------------------------------- REVEL
rv = pd.read_csv(RAW + 'revel-v1.3.SLC19A2_SLC19A3.csv', dtype={'chr': str, 'grch38_pos': str})
rv = rv[rv.grch38_pos != '.'].copy()
rv['k'] = rv.chr + ':' + rv.grch38_pos + '_' + rv.ref + '_' + rv.alt
rv_pick = {}
for k, g in rv.groupby('k'):
    canon = g[g.Ensembl_transcriptid.str.contains(CANON['SLC19A2'] + '|' + CANON['SLC19A3'])]
    if len(canon):
        best = canon.loc[canon.REVEL.idxmax()]
        rule = 'canonical' if len(canon) == 1 else 'canonical_max_of_%d' % len(canon)
    else:
        best = g.loc[g.REVEL.idxmax()]
        rule = 'noncanonical_max_of_%d' % len(g)
    rv_pick[k] = dict(revel=best.REVEL, revel_transcripts=best.Ensembl_transcriptid,
                      revel_aa=f'{best.aaref}>{best.aaalt}', revel_rule=rule, revel_n_rows=len(g))
rvp = pd.DataFrame.from_dict(rv_pick, orient='index')
for c in rvp.columns:
    V[c] = V.k.map(rvp[c])

# ----------------------------------------------------------------- CADD
cadd = {}
for line in open(RAW + 'cadd_GRCh38-v1.7_inclAnno_api_responses.jsonl'):
    r = json.loads(line)
    cadd[r['query']] = r
cad_rows = []
for k in V.k:
    r = cadd.get(k)
    d = dict(k=k)
    if r is None:
        d['cadd_na_reason'] = 'not_queried'
    elif r['status'] != 200:
        d['cadd_na_reason'] = f"http_{r['status']}"
    elif not r['response']:
        d['cadd_na_reason'] = 'api_returned_empty(indel/MNV not scored by API)'
    elif not [x for x in r['response'] if (x['Pos'], x['Ref'], x['Alt']) == (k.split(':')[1].split('_')[0], k.split('_')[1], k.split('_')[2])]:
        # for 1-base-REF insertions the API ignores the inserted sequence and returns the 3 SNVs at the anchor base
        d['cadd_na_reason'] = 'api_returned_only_other_alleles(indel not scored by API)'
    else:
        p0, r0, a0 = k.split(':')[1].split('_')[0], k.split('_')[1], k.split('_')[2]
        resp = [x for x in r['response'] if (x['Pos'], x['Ref'], x['Alt']) == (p0, r0, a0)]
        ph = {x['PHRED'] for x in resp}
        rw = {x['RawScore'] for x in resp}
        assert len(ph) == 1 and len(rw) == 1, (k, ph, rw)
        assert all(x['Ref'] == k.split('_')[1] for x in resp), k
        d['cadd_phred'] = float(ph.pop())
        d['cadd_raw'] = float(rw.pop())
        gene_rows = [x for x in resp if x.get('GeneName') in ('SLC19A2', 'SLC19A3')]
        d['cadd_consequence'] = ';'.join(sorted({x['Consequence'] for x in gene_rows})) or ';'.join(sorted({x['Consequence'] for x in resp}))
        d['cadd_feature'] = ';'.join(sorted({x['FeatureID'] for x in gene_rows}))
        mis = [x for x in gene_rows if x['Consequence'] == 'NON_SYNONYMOUS']
        d['cadd_missense'] = bool(mis)
        d['cadd_aa'] = ';'.join(sorted({f"{x['oAA']}{x['protPos']}{x['nAA']}" for x in mis}))
    cad_rows.append(d)
cad = pd.DataFrame(cad_rows).set_index('k')
for c in ['cadd_phred', 'cadd_raw', 'cadd_consequence', 'cadd_feature', 'cadd_missense', 'cadd_aa', 'cadd_na_reason']:
    V[c] = V.k.map(cad[c]) if c in cad else np.nan

# ----------------------------------------------------------------- missense definition + checks
V['am_scored'] = V.alphamissense.notna()
print('\n=== missense annotation cross-check (SNVs) ===')
snv = V[V.vtype == 'SNV']
print(pd.crosstab([snv.set, snv.am_scored], snv.cadd_missense.fillna('NA'), dropna=False))
ts = V.set == 'timesplit'
print('timesplit is_missense (ClinVar) vs AM-scored:')
print(pd.crosstab(V[ts].is_missense_source, V[ts].am_scored))
mism = V[(V.vtype == 'SNV') & (V.am_scored != V.cadd_missense.fillna(False))]
print('AM vs CADD missense discordant:', len(mism))
print(mism[['set', 'variant_key', 'consequence', 'cadd_consequence', 'cadd_feature', 'am_protein_variant']].to_string())
# AA agreement AM vs CADD (CADD uses Ensembl 110 canonical; for SLC19A3 that is MANE ENST00000644224)
both = V[V.am_scored & V.cadd_missense.fillna(False)]
agree = (both.am_protein_variant == both.cadd_aa)
print('AM protein_variant == CADD oAA/protPos/nAA:', agree.sum(), '/', len(both))
print(both.loc[~agree, ['variant_key', 'am_transcript', 'am_protein_variant', 'cadd_feature', 'cadd_aa']].to_string())
print('CADD features by gene:', V[V.cadd_feature.notna()].groupby('gene').cadd_feature.unique().to_dict())
# REVEL vs AM
print('REVEL rule counts:', V.revel_rule.value_counts(dropna=False).to_dict())
print('REVEL scored but not AM-scored:')
print(V[V.revel.notna() & ~V.am_scored][['set', 'variant_key', 'consequence', 'cadd_consequence', 'revel', 'revel_transcripts', 'revel_rule']].to_string())
print('AM scored but no REVEL:')
print(V[V.am_scored & V.revel.isna()][['set', 'variant_key', 'consequence', 'am_protein_variant']].to_string())
rva = V[V.am_scored & V.revel.notna()]
print('REVEL aa == AM aa (ref/alt AA):', (rva.revel_aa == rva.am_protein_variant.str[0] + '>' + rva.am_protein_variant.str[-1]).sum(), '/', len(rva))
# UniProt/labeled protein sequence check of AM reference residues
lab = pd.read_csv(R + 'data/processed/labeled.csv', usecols=['gene', 'protein_sequence'])
seqs = lab.groupby('gene').protein_sequence.agg(lambda s: s.value_counts().index[0]).to_dict()
okaa = [seqs[g][int(pv[1:-1]) - 1] == pv[0] for g, pv in zip(V[V.am_scored].gene, V[V.am_scored].am_protein_variant)]
print('AM ref residue matches TripleVar protein sequence:', sum(okaa), '/', len(okaa))

# Final missense flag: SNV that is missense in the canonical transcript == present in AM hg38 (canonical) table.
V['is_missense'] = V.am_scored & (V.vtype == 'SNV')

# ----------------------------------------------------------------- write scores table
cols = ['set', 'variant_key', 'gene', 'chrom', 'pos', 'ref', 'alt', 'vtype', 'consequence', 'is_missense', 'label_binary',
        'trusted', 'triplevar_score', 'triplevar_call', 'triplevar_abstract_score', 'triplevar_abstract_call',
        'alphamissense', 'alphamissense_class', 'revel', 'cadd_phred', 'cadd_raw',
        'snp_id', 'y', 'label_source', 'orig_repr', 'feature_source', 'is_missense_source',
        'am_transcript', 'am_protein_variant', 'revel_transcripts', 'revel_aa', 'revel_rule',
        'cadd_consequence', 'cadd_feature', 'cadd_aa', 'cadd_na_reason']
S = V[cols].copy()
S.loc[S.set == 'timesplit', 'trusted'] = np.nan
S.to_csv(OUTD + 'benchmark_scores.csv', index=False)
print('wrote benchmark_scores.csv', S.shape)

# ----------------------------------------------------------------- metrics helpers
TOOLS = {'triplevar': 'triplevar_score', 'triplevar_abstract': 'triplevar_abstract_score',
         'alphamissense': 'alphamissense', 'revel': 'revel', 'cadd': 'cadd_phred'}


def boot_auc(y, s, seed):
    rng = np.random.default_rng(seed)
    n, a, p, tries = len(y), [], [], 0
    while len(a) < N_BOOT and tries < 20 * N_BOOT:
        tries += 1
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        a.append(roc_auc_score(y[i], s[i]))
        p.append(average_precision_score(y[i], s[i]))
    return np.array(a), np.array(p)


def delong_paired(y, s1, s2):
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def plc(s):
        X, Y = s[pos], s[neg]
        v10 = ((Y[None, :] < X[:, None]) + 0.5 * (Y[None, :] == X[:, None])).mean(1)
        v01 = ((X[None, :] > Y[:, None]) + 0.5 * (X[None, :] == Y[:, None])).mean(1)
        return v10, v01

    a10, a01 = plc(s1)
    b10, b01 = plc(s2)
    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    d = a10.mean() - b10.mean()
    if var <= 0:
        return d, np.nan
    return d, float(2 * norm.sf(abs(d) / np.sqrt(var)))


def thr(y, call):
    y, call = np.asarray(y, bool), np.asarray(call, bool)
    tp, fn = int((y & call).sum()), int((y & ~call).sum())
    fp, tn = int((~y & call).sum()), int((~y & ~call).sum())
    f = lambda a, b: a / b if b else np.nan  # noqa: E731
    return dict(tp=tp, fn=fn, fp=fp, tn=tn, accuracy=f(tp + tn, len(y)), sensitivity=f(tp, tp + fn),
                specificity=f(tn, tn + fp), ppv=f(tp, tp + fp), npv=f(tn, tn + fn))


def calls(tool, d):
    """List of (rule_name, mask_of_rows_used, binary_call) at each tool's published threshold."""
    if tool == 'triplevar':
        return [('own_call(grouped_argmax)', np.ones(len(d), bool), d.triplevar_call.astype(bool).values)]
    if tool == 'triplevar_abstract':
        return [('own_call(grouped_argmax)', np.ones(len(d), bool), d.triplevar_abstract_call.astype(bool).values)]
    if tool == 'alphamissense':
        s = d.alphamissense.values
        amb = (s >= 0.34) & (s <= 0.564)
        return [('am>0.564;ambiguous_as_benign', np.ones(len(d), bool), s > 0.564),
                ('am>0.564_vs_<0.34;ambiguous_excluded', ~amb, s > 0.564)]
    if tool == 'revel':
        return [('revel>0.5', np.ones(len(d), bool), d.revel.values > 0.5)]
    if tool == 'cadd':
        return [('cadd_phred>=20', np.ones(len(d), bool), d.cadd_phred.values >= 20)]


METRICS = []


def add(**kw):
    METRICS.append(kw)


def evaluate(cohort, d, tool, scope, seed):
    col = TOOLS[tool]
    d = d[d[col].notna()]
    y = d.label_binary.values.astype(int)
    s = d[col].values.astype(float)
    base = dict(cohort=cohort, scope=scope, tool=tool, n=len(d), n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()))
    if len(d) == 0:
        return None
    if 0 < y.sum() < len(y):
        auc, ap = roc_auc_score(y, s), average_precision_score(y, s)
        ba, bp = boot_auc(y, s, seed)
        add(**base, metric='roc_auc', value=auc, ci_low=np.percentile(ba, 2.5), ci_high=np.percentile(ba, 97.5), rule='', n_boot=len(ba))
        add(**base, metric='pr_auc', value=ap, ci_low=np.percentile(bp, 2.5), ci_high=np.percentile(bp, 97.5), rule='average_precision', n_boot=len(bp))
        add(**base, metric='pr_auc_baseline(prevalence)', value=y.mean(), ci_low=np.nan, ci_high=np.nan, rule='', n_boot=0)
    for rule, mask, c in calls(tool, d):
        t = thr(y[mask], c[mask])
        b2 = dict(base, n=int(mask.sum()), n_pos=int(y[mask].sum()), n_neg=int((1 - y[mask]).sum()))
        for m in ['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv']:
            add(**b2, metric=m, value=t[m], ci_low=np.nan, ci_high=np.nan, rule=rule, n_boot=0,
                tp=t['tp'], fn=t['fn'], fp=t['fp'], tn=t['tn'], n_excluded=int((~mask).sum()))
    if not (0 < y.sum() < len(y)):
        return None
    return dict(base, y=y, s=s)


def paired(cohort, d, focal, other, scope, seed):
    c1, c2 = TOOLS[focal], TOOLS[other]
    d = d[d[c1].notna() & d[c2].notna()]
    y = d.label_binary.values.astype(int)
    if not (0 < y.sum() < len(y)):
        return
    s1, s2 = d[c1].values.astype(float), d[c2].values.astype(float)
    a1, a2 = roc_auc_score(y, s1), roc_auc_score(y, s2)
    rng = np.random.default_rng(seed)
    diffs, tries = [], 0
    while len(diffs) < N_BOOT and tries < 20 * N_BOOT:
        tries += 1
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    diffs = np.array(diffs)
    p_boot = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    p_boot = max(p_boot, 1 / len(diffs))
    _, p_dl = delong_paired(y, s1, s2)
    add(cohort=cohort, scope=scope, tool=f'{focal}_minus_{other}', n=len(d), n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()),
        metric='delta_roc_auc', value=a1 - a2, ci_low=np.percentile(diffs, 2.5), ci_high=np.percentile(diffs, 97.5),
        rule=f'auc_{focal}={a1:.6f};auc_{other}={a2:.6f}', n_boot=len(diffs), p_boot=p_boot, p_delong=p_dl)


# ----------------------------------------------------------------- cohorts
cohorts = []
tsd = V[V.set == 'timesplit']
cvd = V[V.set == 'cv']
cvt = cvd[cvd.trusted.astype(bool)]
for name, d, focal_models in [('timesplit', tsd, ['triplevar', 'triplevar_abstract']),
                              ('cv_all_labels', cvd, ['triplevar']),
                              ('cv_trusted_labels', cvt, ['triplevar'])]:
    cohorts.append((name + '|all_variants', d, focal_models, ['cadd']))
    cohorts.append((name + '|missense', d[d.is_missense], focal_models, ['alphamissense', 'revel', 'cadd']))

ROC = {}
for ci, (cohort, d, focal_models, others) in enumerate(cohorts):
    tools = focal_models + ['alphamissense', 'revel', 'cadd']
    comp = focal_models + others
    common = d.dropna(subset=[TOOLS[t] for t in comp])
    for ti, t in enumerate(tools):
        evaluate(cohort, d, t, 'tool_scored', seed=100 * ci + ti)
    for ti, t in enumerate(comp):
        r = evaluate(cohort, common, t, 'common:' + '+'.join(comp), seed=100 * ci + 50 + ti)
        if r:
            ROC[(cohort, t)] = r
    for fi, f in enumerate(focal_models):
        for oi, o in enumerate(others):
            paired(cohort, d, f, o, 'pairwise_common', seed=10000 + 100 * ci + 10 * fi + oi)
            paired(cohort, common, f, o, 'common:' + '+'.join(comp), seed=20000 + 100 * ci + 10 * fi + oi)
    if 'triplevar_abstract' in focal_models:
        paired(cohort, common, 'triplevar', 'triplevar_abstract', 'common:' + '+'.join(comp), seed=30000 + ci)

M = pd.DataFrame(METRICS)
order = ['cohort', 'scope', 'tool', 'metric', 'rule', 'value', 'ci_low', 'ci_high', 'n', 'n_pos', 'n_neg', 'n_excluded',
         'tp', 'fn', 'fp', 'tn', 'n_boot', 'p_boot', 'p_delong']
M = M[[c for c in order if c in M.columns]]
M.to_csv(OUTD + 'benchmark_metrics.csv', index=False)
print('wrote benchmark_metrics.csv', M.shape)

# ----------------------------------------------------------------- console summary
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_colwidth', 60)
A = M[M.metric.isin(['roc_auc', 'pr_auc'])].copy()
print(A[['cohort', 'scope', 'tool', 'metric', 'value', 'ci_low', 'ci_high', 'n', 'n_pos', 'n_neg']].round(3).to_string())
P = M[M.metric == 'delta_roc_auc']
print(P[['cohort', 'scope', 'tool', 'rule', 'value', 'ci_low', 'ci_high', 'n', 'n_pos', 'n_neg', 'p_boot', 'p_delong']].round(4).to_string())
T = M[M.metric.isin(['accuracy', 'sensitivity', 'specificity'])]
T = T.pivot_table(index=['cohort', 'scope', 'tool', 'rule', 'n', 'n_excluded'], columns='metric', values='value').reset_index()
print(T.round(3).to_string())

# coverage table
cov = []
for setname, d in [('timesplit', tsd), ('cv_all_labels', cvd), ('cv_trusted_labels', cvt)]:
    for t, c in TOOLS.items():
        if t == 'triplevar_abstract' and setname != 'timesplit':
            continue
        cov.append(dict(set=setname, tool=t, n_total=len(d), n_scored=int(d[c].notna().sum()),
                        n_snv=int((d.vtype == 'SNV').sum()), n_missense=int(d.is_missense.sum()),
                        n_missense_scored=int(d[d.is_missense][c].notna().sum())))
COV = pd.DataFrame(cov)
COV.to_csv(OUTD + '_coverage.csv', index=False)
print(COV.to_string())

# ----------------------------------------------------------------- unscored listing
print('\nCADD NA reasons:', V.groupby(['set', 'vtype']).cadd_na_reason.value_counts(dropna=False).to_dict())
print(V[V.cadd_phred.isna()].groupby(['set', 'vtype']).size())

# temporal check for circularity (time-split)
tsx = pd.read_csv(R + 'reports/predictions/clinvar_timesplit.triplevar_v1.csv')
fc = pd.to_datetime(tsx.first_concordant_scv_created, errors='coerce')
dc = pd.to_datetime(tsx.date_created, errors='coerce')
for lbl, cut in [('REVEL v1.3 (2021-05-03)', '2021-05-03'), ('AlphaMissense (2023-08-03)', '2023-08-03'), ('CADD v1.7 release (2023-09?)', '2023-09-01')]:
    print(lbl, 'first concordant SCV before:', int((fc < cut).sum()), '| VCV date_created before:', int((dc < cut).sum()))
    print(tsx.loc[(fc < cut) | (dc < cut), ['variant_id', 'gene', 'hgvs_c', 'consequence', 'clinvar_classification', 'label_binary', 'date_created', 'first_concordant_scv_created']].to_string())
mis_ids = set(V[(V.set == 'timesplit') & V.is_missense].variant_key)
tsm = tsx[tsx.variant_id.isin(mis_ids)]
print(tsm[['variant_id', 'gene', 'hgvs_c', 'hgvs_p', 'clinvar_classification', 'review_stars', 'label_binary', 'date_created', 'first_concordant_scv_created', 'classification_history']].to_string())
xm = V[(V.set == 'timesplit') & V.is_missense][['variant_key', 'label_binary', 'triplevar_score', 'triplevar_abstract_score', 'alphamissense', 'alphamissense_class', 'revel', 'cadd_phred']]
print(xm.round(3).to_string())
print('CADD-unscored SNVs:')
print(V[(V.vtype == 'SNV') & V.cadd_phred.isna()][['set', 'variant_key', 'cadd_na_reason']].to_string())

import pickle  # noqa: E402
pickle.dump({k: (v['y'], v['s'], v['n'], v['n_pos'], v['n_neg']) for k, v in ROC.items()}, open(sys.argv[2], 'wb'))
