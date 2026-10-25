"""Descriptive check of the trust experiment data (Assignment 3, Segment 4).
Run from the repo root:  python analysis/analysis.py
Reads data/raw/trust_experiment_data.csv, writes data/processed/clean_data.csv
and analysis/summary_by_condition.csv. Raw file is never edited; the only rows
removed are pilot rows (pilot = 1)."""
import pandas as pd, numpy as np
from scipy import stats

raw = pd.read_csv('data/raw/trust_experiment_data.csv', encoding='utf-8-sig')
d = raw[raw.pilot == 0].copy()                          # drop pilot rows only
d['short_session'] = (d.duration_s < 120).astype(int)   # flag only, nothing else removed
d.to_csv('data/processed/clean_data.csv', index=False)
print(f'rows: raw {len(raw)} -> clean {len(d)} (pilot removed: {raw[raw.pilot == 1].pid.tolist()})\n')

pts = [f'pts{i}' for i in range(1, 11)]
assert (d[pts].sum(axis=1) == d.trust_total).all(), 'totals do not recompute'

g = d.groupby('condition').trust_total.agg(['count', 'mean', 'std', 'min', 'max']).round(2)
g.to_csv('analysis/summary_by_condition.csv'); print(g, '\n')

cor, inc = d[d.accuracy == 'correct'].trust_total, d[d.accuracy == 'incorrect'].trust_total
diff = cor.mean() - inc.mean()
se = np.sqrt(cor.var(ddof=1)/len(cor) + inc.var(ddof=1)/len(inc))
t = stats.ttest_ind(cor, inc, equal_var=False)
print(f'ER1 correct-incorrect = {diff:.2f}  95% CI {diff-1.96*se:.1f} to {diff+1.96*se:.1f}  Welch p={t.pvalue:.6f}')

m = d.groupby('condition').trust_total.mean()
drop_s, drop_c = m['A']-m['B'], m['C']-m['D']
se_i = np.sqrt(sum(d[d.condition == c].trust_total.var(ddof=1)/(d.condition == c).sum() for c in 'ABCD'))
print(f'ER2 drop simple={drop_s:.2f} drop complex={drop_c:.2f} interaction={drop_c-drop_s:.2f}  CI {drop_c-drop_s-1.96*se_i:.1f} to {drop_c-drop_s+1.96*se_i:.1f}')
print('ER3 mean A =', round(m['A'], 2), '| ER4 mean B =', round(m['B'], 2), ', D =', round(m['D'], 2))
print('ER6 check_correct by accuracy', d.groupby('accuracy').check_correct.mean().round(2).to_dict())
print('ER7 check_difficulty by difficulty', d.groupby('difficulty').check_difficulty.mean().round(2).to_dict())
print('ER8 realism >= 4:', int((d.realism >= 4).sum()), 'of', len(d))
print('ER9 r(prior_use, trust_total) = %.2f (p=%.2f)' % stats.pearsonr(d.prior_use, d.trust_total))
k = 10; it = d[pts]
alpha = k/(k-1)*(1 - it.var(ddof=1).sum()/it.sum(axis=1).var(ddof=1))
print('ER10 Cronbach alpha =', round(alpha, 2))
print('duration median (min):', round(d.duration_s.median()/60, 1), '| under 2 min:', d[d.short_session == 1].pid.tolist())
print('without short sessions:', d[d.short_session == 0].groupby('condition').trust_total.mean().round(2).to_dict())
