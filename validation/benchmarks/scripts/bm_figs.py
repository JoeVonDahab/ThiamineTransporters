"""ROC figures on the common subsets (one line per tool)."""
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

OUTD = '/home/joe/projects/TripleVar/validation/benchmarks/'
ROC = pickle.load(open(sys.argv[1], 'rb'))
M = pd.read_csv(OUTD + 'benchmark_metrics.csv')

INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
STYLE = {  # fixed colour per tool (reference categorical order), secondary encoding by dash for the abstract model
    'triplevar': dict(label='TripleVar v1', color='#2a78d6', ls='-'),
    'cadd': dict(label='CADD v1.7', color='#eb6834', ls='-'),
    'alphamissense': dict(label='AlphaMissense', color='#1baf7a', ls='-'),
    'revel': dict(label='REVEL v1.3', color='#eda100', ls='-'),
    'triplevar_abstract': dict(label='TripleVar (abstract model)', color='#e87ba4', ls=(0, (4, 2))),
}
ORDER = ['triplevar', 'triplevar_abstract', 'alphamissense', 'revel', 'cadd']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
                     'axes.linewidth': 0.8, 'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def auc_row(cohort, tool):
    r = M[(M.cohort == cohort) & (M.tool == tool) & (M.metric == 'roc_auc') & M.scope.str.startswith('common')]
    return r.iloc[0]


def panel(ax, cohort, desc, letter):
    tools = [t for t in ORDER if (cohort, t) in ROC]
    n = npos = nneg = None
    for t in tools:
        y, s, n, npos, nneg = ROC[(cohort, t)]
        fpr, tpr, _ = roc_curve(y, s)
        r = auc_row(cohort, t)
        st = STYLE[t]
        ax.plot(fpr, tpr, color=st['color'], ls=st['ls'], lw=1.6, solid_capstyle='round', solid_joinstyle='round',
                label=f"{st['label']}  {r.value:.3f} [{r.ci_low:.2f}–{r.ci_high:.2f}]  n={int(r.n)}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=(0, (1, 2)), zorder=0)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01); ax.set_aspect('equal')
    ax.set_xticks(np.linspace(0, 1, 6)); ax.set_yticks(np.linspace(0, 1, 6))
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.set_xlabel('False positive rate (1 − specificity)'); ax.set_ylabel('True positive rate (sensitivity)')
    leg = ax.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.17), frameon=True, fontsize=6.5,
                    title=f'{desc}\nn = {n} ({npos} P/LP, {nneg} B/LB)  •  ROC-AUC [95% CI]',
                    title_fontsize=6.5, handlelength=2.4, borderpad=0.6, labelspacing=0.35)
    leg.get_frame().set_edgecolor(GRID); leg.get_frame().set_linewidth(0.6); leg.get_frame().set_facecolor('white')
    leg.get_title().set_color(INK2); leg._legend_box.align = 'left'
    for txt in leg.get_texts():
        txt.set_color(INK)
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight='bold', color=INK, va='bottom')


def fig(panels, out):
    f, axs = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 4.6))
    axs = np.atleast_1d(axs)
    for ax, (cohort, desc, letter) in zip(axs, panels):
        panel(ax, cohort, desc, letter)
    f.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.31, wspace=0.35)
    f.savefig(out, dpi=300)
    print('wrote', out)


fig([('timesplit|all_variants', 'ClinVar time-split: all SNVs (every consequence; CADD-scorable)', 'a'),
     ('timesplit|missense', 'ClinVar time-split: missense SNVs (small n)', 'b')], OUTD + 'fig_roc_timesplit.png')
fig([('cv_all_labels|missense', 'CV set, missense SNVs, all labels', 'a'),
     ('cv_trusted_labels|missense', 'CV set, missense SNVs, trusted labels', 'b')], OUTD + 'fig_roc_cv_missense.png')
