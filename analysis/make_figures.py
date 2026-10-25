"""Makes Figure 8 and Figure 9 of Assignment 3 from the raw data file.
Run from the repo root:  python analysis/make_figures.py
Needs: pandas, numpy, matplotlib.  Uses only main sessions (pilot = 0).
Output: analysis/figures/fig8_means.png and analysis/figures/fig9_difficulty.png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = pd.read_csv('data/raw/trust_experiment_data.csv')
m = d[d.pilot == 0]
BLUE, ORANGE = '#2a6fbb', '#c8590e'

# Figure 8: mean trust total by condition with 95% intervals
fig, ax = plt.subplots(figsize=(6, 3.8), dpi=200)
for diff, lab, col, mk in (('simple', 'Simple questions (A to B)', BLUE, 'o'),
                           ('complex', 'Complex questions (C to D)', ORANGE, 's')):
    ms, es = [], []
    for acc in ('correct', 'incorrect'):
        v = m[(m.difficulty == diff) & (m.accuracy == acc)].trust_total
        ms.append(v.mean()); es.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
    ax.errorbar([0, 1], ms, yerr=es, color=col, marker=mk, ms=8, lw=2, capsize=4,
                mec='white', mew=1.5, label=lab)
    ax.annotate(f'{ms[0]:.1f}', (0, ms[0]), textcoords='offset points', xytext=(-26, -3), fontsize=9)
    ax.annotate(f'{ms[1]:.1f}', (1, ms[1]), textcoords='offset points', xytext=(10, -3), fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Correct reply', 'Incorrect reply'])
ax.set_xlim(-0.4, 1.4); ax.set_ylim(0, 40); ax.set_ylabel('Trust total (0-40)')
ax.axhspan(27, 40, color='#e8f1e4', zorder=0); ax.text(1.38, 36, 'trust band', ha='right', fontsize=8, color='#555')
ax.axhspan(0, 13, color='#f6e6e1', zorder=0); ax.text(1.38, 1.5, 'distrust band', ha='right', fontsize=8, color='#555')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.0, 0.2), fontsize=9)
ax.set_title('Mean trust by condition (main sessions, 10 to 11 per point, 95% interval)', fontsize=9.5, loc='left')
fig.tight_layout(); fig.savefig('analysis/figures/fig8_means.png'); plt.close(fig)

# Figure 9: difficulty as rated by participants (check question, 1-7)
rng = np.random.default_rng(1)
fig, ax = plt.subplots(figsize=(6, 3.8), dpi=200)
for i, (k, col) in enumerate((('simple', BLUE), ('complex', ORANGE))):
    v = m[m.difficulty == k].check_difficulty.values
    ax.scatter(i + rng.uniform(-.18, .18, len(v)), v + rng.uniform(-.12, .12, len(v)),
               s=28, color=col, alpha=.55, edgecolor='white', linewidth=.5)
    ax.scatter([i], [v.mean()], s=110, color=col, marker='D', edgecolor='black', linewidth=1.2, zorder=3)
    ax.annotate(f'mean {v.mean():.2f} (n = {len(v)})', (i, v.mean()), textcoords='offset points', xytext=(18, -3), fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Simple questions (A, B)', 'Complex questions (C, D)'])
ax.set_xlim(-0.6, 1.9); ax.set_ylim(0.5, 7.5); ax.set_yticks(range(1, 8))
ax.set_ylabel('"How difficult were the situations?" (1-7)')
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_title('Difficulty as rated by participants (main sessions)', fontsize=10, loc='left')
fig.tight_layout(); fig.savefig('analysis/figures/fig9_difficulty.png'); plt.close(fig)
print('saved analysis/figures/fig8_means.png and fig9_difficulty.png')
