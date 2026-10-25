"""Check of the trust experiment data (Assignment 3, Segment 4).
Run from the repo root:  python analysis/analysis.py
Reads data/raw/trust_experiment_data.csv (version 3, see cleaning_log.txt), uses the pilot column,
writes data/processed/clean_data.csv and analysis/summary_by_condition.csv.
Pilot sessions are not used in the main analysis."""
import pandas as pd, numpy as np
from scipy import stats



raw = pd.read_csv('data/raw/trust_experiment_data.csv')
d = raw.copy()

d.to_csv('data/processed/clean_data.csv', index=False)

pts = [f'pts{i}' for i in range(1, 11)]
assert (d[pts].sum(axis=1) == d.trust_total).all(), 'totals do not recompute'
m = d[d.pilot == 0]
print('rows:', len(d), '| pilot:', int(d.pilot.sum()), '| main:', len(m), m.condition.value_counts().sort_index().to_dict(), '\n')

g = m.groupby('condition').trust_total.agg(['count', 'mean', 'std', 'min', 'max']).round(2)
g.to_csv('analysis/summary_by_condition.csv'); print(g, '\n')

cor, inc = m[m.accuracy == 'correct'].trust_total, m[m.accuracy == 'incorrect'].trust_total
diff = cor.mean() - inc.mean()
se = np.sqrt(cor.var(ddof=1)/len(cor) + inc.var(ddof=1)/len(inc))
t = stats.ttest_ind(cor, inc, equal_var=False)
print(f'ER1 correct-incorrect = {diff:.2f}  95% CI {diff-1.96*se:.1f} to {diff+1.96*se:.1f}  Welch p={t.pvalue:.6f}')

mu = m.groupby('condition').trust_total.mean(); n = m.groupby('condition').size()
drop_s, drop_c = mu['A']-mu['B'], mu['C']-mu['D']
se_i = np.sqrt(sum(m[m.condition == c].trust_total.var(ddof=1)/n[c] for c in 'ABCD'))
print(f'ER2 drop simple={drop_s:.2f} drop complex={drop_c:.2f} interaction={drop_c-drop_s:.2f}  CI {drop_c-drop_s-1.96*se_i:.1f} to {drop_c-drop_s+1.96*se_i:.1f}')
print('ER3 mean A =', round(mu['A'], 2), '| ER4 mean B =', round(mu['B'], 2), ', D =', round(mu['D'], 2))
print('ER6 check_correct by accuracy', m.groupby('accuracy').check_correct.mean().round(2).to_dict())
print('ER7 check_difficulty by difficulty', m.groupby('difficulty').check_difficulty.mean().round(2).to_dict())
print('ER8 realism >= 4:', int((m.realism >= 4).sum()), 'of', len(m), '=', round((m.realism >= 4).mean()*100, 1), 'percent')
print('ER9 r(prior_use, trust_total) = %.2f (p=%.2f)' % stats.pearsonr(m.prior_use, m.trust_total))
k = 10; it = m[pts]
print('ER10 Cronbach alpha =', round(k/(k-1)*(1 - it.var(ddof=1).sum()/it.sum(axis=1).var(ddof=1)), 2))
print('duration (min) median', round(m.duration_s.median()/60, 1), '| min', round(m.duration_s.min()/60, 1), '| max', round(m.duration_s.max()/60, 1))
