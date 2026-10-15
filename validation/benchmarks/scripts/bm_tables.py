"""Render the key benchmark tables as markdown fragments (written to argv[1])."""
import sys

import numpy as np
import pandas as pd

OUTD = '/home/joe/projects/TripleVar/validation/benchmarks/'
M = pd.read_csv(OUTD + 'benchmark_metrics.csv')
COV = pd.read_csv(OUTD + '_coverage.csv')
NAME = {'triplevar': 'TripleVar v1', 'triplevar_abstract': 'TripleVar abstract model', 'alphamissense': 'AlphaMissense',
        'revel': 'REVEL v1.3', 'cadd': 'CADD v1.7 (PHRED)'}
SETNAME = {'timesplit': 'ClinVar time-split', 'cv_all_labels': 'CV, all labels', 'cv_trusted_labels': 'CV, trusted labels'}
ORDER = ['timesplit|all_variants', 'timesplit|missense', 'cv_all_labels|all_variants', 'cv_all_labels|missense',
         'cv_trusted_labels|all_variants', 'cv_trusted_labels|missense']
TORDER = ['triplevar', 'triplevar_abstract', 'alphamissense', 'revel', 'cadd']


def subset(cohort, scope):
    s, sub = cohort.split('|')
    if sub == 'missense':
        return SETNAME[s], 'missense SNVs'
    if scope == 'tool_scored':
        return SETNAME[s], 'all variants (incl. indels)'
    return SETNAME[s], 'all SNVs (any consequence)'


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '|'.join(['---'] * len(cols)) + '|']
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join('' if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + ' |')
    return '\n'.join(out)


def f3(x):
    return '' if pd.isna(x) else f'{x:.3f}'


frag = {}
c = COV.copy()
c['set'] = c.set.map(SETNAME)
c['tool'] = c.tool.map(NAME)
frag['coverage'] = md(c.rename(columns={'n_total': 'variants', 'n_scored': 'scored', 'n_snv': 'SNVs',
                                        'n_missense': 'missense SNVs', 'n_missense_scored': 'missense scored'}))

A = M[M.metric == 'roc_auc'].copy()
keep = A.scope.str.startswith('common') | ((A.scope == 'tool_scored') & A.cohort.str.endswith('all_variants') & A.tool.isin(['triplevar', 'triplevar_abstract']))
A = A[keep].copy()
A['o1'] = A.cohort.map(ORDER.index)
A['o2'] = (A.scope != 'tool_scored').astype(int)
A['o3'] = A.tool.map(TORDER.index)
A = A.sort_values(['o1', 'o2', 'o3'])
P = M[M.metric == 'pr_auc'].set_index(['cohort', 'scope', 'tool'])
rows = []
for _, r in A.iterrows():
    p = P.loc[(r.cohort, r.scope, r.tool)]
    s, sub = subset(r.cohort, r.scope)
    rows.append({'set': s, 'variants': sub, 'tool': NAME[r.tool], 'n (P/LP:B/LB)': f'{int(r.n)} ({int(r.n_pos)}:{int(r.n_neg)})',
                 'ROC-AUC [95% CI]': f'{r.value:.3f} [{r.ci_low:.3f}-{r.ci_high:.3f}]',
                 'PR-AUC [95% CI]': f'{p.value:.3f} [{p.ci_low:.3f}-{p.ci_high:.3f}]', 'PR-AUC baseline': f'{r.n_pos / r.n:.3f}'})
frag['auc'] = md(pd.DataFrame(rows))

D = M[(M.metric == 'delta_roc_auc') & M.scope.str.startswith('common')].copy()
D['o1'] = D.cohort.map(ORDER.index)
D = D.sort_values('o1')
rows = []
for _, r in D.iterrows():
    f, o = r.tool.split('_minus_')
    au = dict(kv.split('=') for kv in r.rule.split(';'))
    s, sub = subset(r.cohort, r.scope)
    rows.append({'set': s, 'variants': sub, 'comparison': f'{NAME[f]} minus {NAME[o]}', 'n (P/LP:B/LB)': f'{int(r.n)} ({int(r.n_pos)}:{int(r.n_neg)})',
                 'AUC (first, second)': f"{float(au['auc_' + f]):.3f}, {float(au['auc_' + o]):.3f}",
                 'delta AUC [95% CI]': f'{r.value:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]',
                 'p bootstrap': f'{r.p_boot:.4f}', 'p DeLong': '' if pd.isna(r.p_delong) else f'{r.p_delong:.2g}'})
frag['paired'] = md(pd.DataFrame(rows))

T = M[M.metric.isin(['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv']) & (M.scope.str.startswith('common') | ((M.scope == 'tool_scored') & M.cohort.str.endswith('all_variants') & M.tool.isin(['triplevar', 'triplevar_abstract'])))]
T = T.pivot_table(index=['cohort', 'scope', 'tool', 'rule', 'n', 'n_pos', 'n_neg', 'n_excluded', 'tp', 'fn', 'fp', 'tn'], columns='metric', values='value').reset_index()
T['o1'] = T.cohort.map(ORDER.index)
T['o2'] = (T.scope != 'tool_scored').astype(int)
T['o3'] = T.tool.map(TORDER.index)
T = T.sort_values(['o1', 'o2', 'o3', 'rule'])
rows = []
for _, r in T.iterrows():
    s, sub = subset(r.cohort, r.scope)
    rows.append({'set': s, 'variants': sub, 'tool': NAME[r.tool], 'rule': r.rule, 'n (P:B)': f'{int(r.n)} ({int(r.n_pos)}:{int(r.n_neg)})',
                 'excl.': int(r.n_excluded), 'TP/FN/FP/TN': f'{int(r.tp)}/{int(r.fn)}/{int(r.fp)}/{int(r.tn)}',
                 'accuracy': f3(r.accuracy), 'sensitivity': f3(r.sensitivity), 'specificity': f3(r.specificity), 'PPV': f3(r.ppv), 'NPV': f3(r.npv)})
frag['thresholds'] = md(pd.DataFrame(rows))

with open(sys.argv[1], 'w') as fh:
    for k, v in frag.items():
        fh.write(f'<<{k}>>\n{v}\n\n')
for k, v in frag.items():
    print(f'### {k}\n{v}\n')
