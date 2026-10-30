"""Assignment 4 analysis of the trust experiment.
Run from the repo root:  python analysis/analysis_a4.py
Reads data/raw/trust_experiment_data.csv (version 3). Main analysis = rows with pilot == 0 (43 sessions).
Writes tables (csv), numbers (a4_numbers.txt) and figures (png) to analysis/a4/.
Every number in the Assignment 4 report comes from a4_numbers.txt or one of the csv tables in that folder."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'analysis/a4'
os.makedirs(OUT, exist_ok=True)
raw = pd.read_csv('data/raw/trust_experiment_data.csv')
m = raw[raw.pilot == 0].copy()
pts = [f'pts{i}' for i in range(1, 11)]
lines = []
def say(s=''):
    print(s); lines.append(s)

BLUE, ORANGE, GREY = '#1f77b4', '#d95f02', '#666666'
COL = {'A': BLUE, 'C': BLUE, 'B': ORANGE, 'D': ORANGE}
LAB = {'A': 'A simple + correct', 'B': 'B simple + incorrect', 'C': 'C complex + correct', 'D': 'D complex + incorrect'}

def welch(a, b):
    """mean(a)-mean(b), Welch 95% CI (t based), p"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    tc = stats.t.ppf(0.975, df)
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    return d, d - tc * se, d + tc * se, p, df

# ---------------- 1. data stages --------------------------------------------------
say('== 1. Rows at each stage ==')
say(f'rows in raw file: {len(raw)}; pilot rows: {int(raw.pilot.sum())}; main rows: {len(m)}')
say('duplicate pid:', ) if False else None
say(f'duplicate pid: {int(raw.pid.duplicated().sum())}')
req = ['condition', 'prior_use', 'trust_total', 'realism', 'check_correct', 'check_difficulty'] + pts
say(f'missing in required columns: {int(raw[req].isna().sum().sum())}')
say(f'totals that do not recompute from pts1-pts10: {int((raw[pts].sum(axis=1) != raw.trust_total).sum())}')
say(f'trust_total outside 0-40: {int(((raw.trust_total < 0) | (raw.trust_total > 40)).sum())}')
say(f'open_text filled (main): {int(m.open_text.notna().sum())} of {len(m)}; guess_text filled (main): {int(m.guess_text.notna().sum())} of {len(m)}')
say(f'manual assignment in main rows: {int((m.assigned_by == "manual").sum())}; in pilot rows: {int((raw[raw.pilot==1].assigned_by == "manual").sum())}')
say('main rows per condition: ' + str(m.condition.value_counts().sort_index().to_dict()))
say(f'incorrect replies word-length rule: see scenarios/scenario_pool.csv')

# ---------------- 2. condition table ---------------------------------------------------
say('\n== 2. Trust total by condition (main sessions) ==')
rows = []
for c in 'ABCD':
    x = m[m.condition == c].trust_total
    q1, q3 = x.quantile(.25), x.quantile(.75)
    rows.append(dict(condition=c, n=len(x), mean=x.mean(), median=x.median(), sd=x.std(ddof=1), q1=q1, q3=q3,
                     min=x.min(), max=x.max(), range=x.max() - x.min(),
                     in_trust_band=int((x >= 27).sum()), in_mixed_band=int(((x >= 14) & (x <= 26)).sum()), in_distrust_band=int((x <= 13).sum())))
T = pd.DataFrame(rows).round(2)
T.to_csv(f'{OUT}/table_condition_summary.csv', index=False)
say(T.to_string(index=False))

# ---------------- 3. difference from baseline -----------------------------------------------
say('\n== 3. Difference from baseline A (points, 0-40 scale) ==')
A = m[m.condition == 'A'].trust_total
rows = []
for c in 'BCD':
    x = m[m.condition == c].trust_total
    d, lo, hi, p, df = welch(x, A)
    mw = stats.mannwhitneyu(x, A, alternative='two-sided').pvalue
    rows.append(dict(condition=c, n=len(x), mean_diff_vs_A=d, ci_low=lo, ci_high=hi, welch_p=p, mannwhitney_p=mw,
                     median_diff_vs_A=x.median() - A.median()))
TD = pd.DataFrame(rows).round(6)
TD.to_csv(f'{OUT}/table_difference_from_baseline.csv', index=False)
say(TD.to_string(index=False))
for r_ in TD.itertuples():
    say(f'{r_.condition} against A: mean difference {r_.mean_diff_vs_A:.2f}, 95% CI {r_.ci_low:.2f} to {r_.ci_high:.2f}, Welch p = {r_.welch_p:.6f}, median difference {r_.median_diff_vs_A:.1f}')

# ---------------- 4. main effect of accuracy ---------------------------------------------
say('\n== 4. Accuracy effect (ER1) ==')
cor, inc = m[m.accuracy == 'correct'].trust_total, m[m.accuracy == 'incorrect'].trust_total
d, lo, hi, p, df = welch(cor, inc)
sp = np.sqrt(((len(cor) - 1) * cor.var(ddof=1) + (len(inc) - 1) * inc.var(ddof=1)) / (len(cor) + len(inc) - 2))
say(f'n correct = {len(cor)}, n incorrect = {len(inc)}')
say(f'mean correct = {cor.mean():.2f}, mean incorrect = {inc.mean():.2f}, median correct = {cor.median():.1f}, median incorrect = {inc.median():.1f}')
say(f'difference = {d:.2f} points, 95% CI {lo:.2f} to {hi:.2f}, Welch p = {p:.6f}, df = {df:.1f}')
say(f'Mann-Whitney p = {stats.mannwhitneyu(cor, inc, alternative="two-sided").pvalue:.6f}; Cohen d = {d/sp:.2f}')
say(f'difference of medians = {cor.median() - inc.median():.1f} points')
say(f'threshold 8 points: observed {d:.2f} -> crossed point estimate: {d >= 8}; lower CI end >= 8: {lo >= 8}')
# bootstrap
rng = np.random.default_rng(2431)
bs = [rng.choice(cor, len(cor)).mean() - rng.choice(inc, len(inc)).mean() for _ in range(10000)]
say(f'bootstrap 95% interval of the difference (10000 resamples, seed 2431): {np.percentile(bs, 2.5):.2f} to {np.percentile(bs, 97.5):.2f}; share of resamples with difference >= 8: {np.mean(np.array(bs) >= 8):.3f}')
er1 = dict(d=d, lo=lo, hi=hi, p=p)

# ---------------- 5. interaction ---------------------------------------------------------
say('\n== 5. Interaction (ER2) ==')
mu = m.groupby('condition').trust_total.mean()
n = m.groupby('condition').size()
var = m.groupby('condition').trust_total.var(ddof=1)
drop_s, drop_c = mu['A'] - mu['B'], mu['C'] - mu['D']
say(f'drop simple (A-B) = {drop_s:.2f}; drop complex (C-D) = {drop_c:.2f}; difference of drops = {drop_c - drop_s:.2f}')
m['acc_inc'] = (m.accuracy == 'incorrect').astype(int)
m['cx'] = (m.difficulty == 'complex').astype(int)
f = smf.ols('trust_total ~ acc_inc * cx', data=m).fit()
ci = f.conf_int().loc['acc_inc:cx']
say(f'OLS interaction coefficient (incorrect x complex) = {f.params["acc_inc:cx"]:.2f}, 95% CI {ci[0]:.2f} to {ci[1]:.2f}, p = {f.pvalues["acc_inc:cx"]:.3f}  (negative = bigger drop for complex)')
say(f'OLS: effect of incorrect in simple = {f.params["acc_inc"]:.2f}; effect of complex when correct = {f.params["cx"]:.2f}')
fm = smf.ols('trust_total ~ acc_inc * cx', data=m).fit()
ci_c = fm.conf_int().loc['cx']
say(f'complexity effect among correct replies (C - A) = {f.params["cx"]:.2f}, 95% CI {ci_c[0]:.2f} to {ci_c[1]:.2f}, p = {f.pvalues["cx"]:.3f}')
f2 = smf.ols('trust_total ~ acc_inc * cx + prior_use + realism', data=m).fit()
ci2 = f2.conf_int()
say(f'with covariates prior_use and realism: accuracy effect = {-f2.params["acc_inc"]:.2f} (simple, CI {-ci2.loc["acc_inc"][1]:.2f} to {-ci2.loc["acc_inc"][0]:.2f}); interaction = {f2.params["acc_inc:cx"]:.2f} (CI {ci2.loc["acc_inc:cx"][0]:.2f} to {ci2.loc["acc_inc:cx"][1]:.2f}, p = {f2.pvalues["acc_inc:cx"]:.3f}); prior_use coef = {f2.params["prior_use"]:.2f} (p = {f2.pvalues["prior_use"]:.3f}); realism coef = {f2.params["realism"]:.2f} (p = {f2.pvalues["realism"]:.3f})')
say(f'power note: smallest interaction this design (n about 11 per cell) detects with 80% power: about {2.8 * np.sqrt(sum(var / n)):.1f} points (2.8 x SE of the interaction, SE = {np.sqrt(sum(var / n)):.2f})')

# main-effect models (no interaction), unadjusted and adjusted
g0 = smf.ols('trust_total ~ acc_inc + cx', data=m).fit()
g1 = smf.ols('trust_total ~ acc_inc + cx + prior_use + realism', data=m).fit()
for nm, g in [('without covariates', g0), ('with prior_use and realism', g1)]:
    c_ = g.conf_int()
    say(f'main-effect model {nm}: accuracy effect (correct minus incorrect) = {-g.params["acc_inc"]:.2f} points, 95% CI {-c_.loc["acc_inc"][1]:.2f} to {-c_.loc["acc_inc"][0]:.2f}, p = {g.pvalues["acc_inc"]:.6f}; complexity effect = {g.params["cx"]:.2f} (CI {c_.loc["cx"][0]:.2f} to {c_.loc["cx"][1]:.2f}, p = {g.pvalues["cx"]:.3f})')
say(f'realism in the adjusted main-effect model: coefficient {g1.params["realism"]:.2f} points per scale step (CI {g1.conf_int().loc["realism"][0]:.2f} to {g1.conf_int().loc["realism"][1]:.2f}, p = {g1.pvalues["realism"]:.3f})')
say(f'2x2 R-squared = {f.rsquared:.3f}')

# ---------------- 6. bands and thresholds ----------------------------------------------------
say('\n== 6. Expected bands (ER3, ER4) ==')
say(f'mean A = {mu["A"]:.2f}, B = {mu["B"]:.2f}, C = {mu["C"]:.2f}, D = {mu["D"]:.2f}')
say(f'A median = {m[m.condition=="A"].trust_total.median():.1f}; sessions in trust band: A {int((A>=27).sum())} of {len(A)}')
for c in 'BD':
    x = m[m.condition == c].trust_total
    say(f'{c}: mean {x.mean():.2f}; in trust band {int((x>=27).sum())} of {len(x)}; in mixed band {int(((x>=14)&(x<=26)).sum())}; in distrust band {int((x<=13).sum())}')
say(f'T10 expected means: A 28-35 -> A {mu["A"]:.1f}; C 26-33 -> C {mu["C"]:.1f}; B 14-22 -> B {mu["B"]:.1f}; D 8-16 -> D {mu["D"]:.1f}')
say(f'C mean {mu["C"]:.2f} vs A mean {mu["A"]:.2f}: difference {mu["A"]-mu["C"]:.2f}')

# ---------------- 7. check questions ----------------------------------------------------------
say('\n== 7. Check questions (ER6, ER7) ==')
cc = m.groupby('condition').check_correct.agg(['mean', 'median'])
say('check_correct (0-3) by condition: ' + '; '.join(f'{c} mean {cc.loc[c,"mean"]:.2f} median {cc.loc[c,"median"]:.1f}' for c in 'ABCD'))
say(f'check_correct by accuracy: correct mean {m[m.accuracy=="correct"].check_correct.mean():.2f}, incorrect mean {m[m.accuracy=="incorrect"].check_correct.mean():.2f}')
miss_inc = m[(m.accuracy == 'incorrect') & (m.check_correct >= 2)]
miss_cor = m[(m.accuracy == 'correct') & (m.check_correct <= 1)]
say(f'incorrect-condition sessions with check_correct >= 2: {len(miss_inc)} of {int((m.accuracy=="incorrect").sum())} ({", ".join(miss_inc.pid)})')
say(f'correct-condition sessions with check_correct <= 1: {len(miss_cor)} of {int((m.accuracy=="correct").sum())} ({", ".join(miss_cor.pid)})')
dd = m.groupby('difficulty').check_difficulty.agg(['mean', 'median'])
say(f'check_difficulty (1-7): simple mean {dd.loc["simple","mean"]:.2f} median {dd.loc["simple","median"]:.1f}; complex mean {dd.loc["complex","mean"]:.2f} median {dd.loc["complex","median"]:.1f}; gap {dd.loc["complex","mean"]-dd.loc["simple","mean"]:.2f}')
gs, gc = m[m.difficulty == 'simple'].check_difficulty, m[m.difficulty == 'complex'].check_difficulty
d2, l2, h2, p2, _ = welch(gc, gs)
say(f'difficulty gap CI {l2:.2f} to {h2:.2f}, Welch p = {p2:.6f}')
say(f'check_difficulty by accuracy: correct {m[m.accuracy=="correct"].check_difficulty.mean():.2f}, incorrect {m[m.accuracy=="incorrect"].check_difficulty.mean():.2f}')
# trust among those who noticed vs not (incorrect conditions)
inc_df = m[m.accuracy == 'incorrect']
nt = inc_df[inc_df.check_correct >= 2].trust_total; yt = inc_df[inc_df.check_correct <= 1].trust_total
say(f'incorrect conditions: noticed (check_correct <= 1) n = {len(yt)}, mean trust {yt.mean():.2f}; did not notice (>= 2) n = {len(nt)}, mean trust {nt.mean():.2f}')
# manipulation-check sensitivity
ok = m[((m.accuracy == 'correct') & (m.check_correct >= 2)) | ((m.accuracy == 'incorrect') & (m.check_correct <= 1))]
c2, i2 = ok[ok.accuracy == 'correct'].trust_total, ok[ok.accuracy == 'incorrect'].trust_total
dd2, ll2, hh2, pp2, _ = welch(c2, i2)
say(f'sensitivity: only sessions where the check question matches the manipulation: n = {len(ok)} ({len(c2)} correct, {len(i2)} incorrect); difference = {dd2:.2f}, CI {ll2:.2f} to {hh2:.2f}, p = {pp2:.6f}')
allr = raw
c3, i3 = allr[allr.accuracy == 'correct'].trust_total, allr[allr.accuracy == 'incorrect'].trust_total
d3, l3, h3, p3, _ = welch(c3, i3)
say(f'sensitivity: all 48 rows including pilot: difference = {d3:.2f}, CI {l3:.2f} to {h3:.2f}, p = {p3:.6f}')

# ---------------- 8. confounders -----------------------------------------------------------------
say('\n== 8. Confounders and balance ==')
say('prior_use (1-5) median by condition: ' + '; '.join(f'{c} {m[m.condition==c].prior_use.median():.1f}' for c in 'ABCD') + f'; Kruskal-Wallis p = {stats.kruskal(*[m[m.condition==c].prior_use for c in "ABCD"]).pvalue:.3f}')
say('realism (1-7) median by condition: ' + '; '.join(f'{c} {m[m.condition==c].realism.median():.1f} (mean {m[m.condition==c].realism.mean():.2f})' for c in 'ABCD') + f'; Kruskal-Wallis p = {stats.kruskal(*[m[m.condition==c].realism for c in "ABCD"]).pvalue:.3f}')
say(f'realism >= 4: {int((m.realism>=4).sum())} of {len(m)} ({(m.realism>=4).mean()*100:.1f} percent); mean realism {m.realism.mean():.2f}; median {m.realism.median():.1f}')
r, pr = stats.pearsonr(m.prior_use, m.trust_total); rs, ps = stats.spearmanr(m.prior_use, m.trust_total)
say(f'prior_use vs trust_total: Pearson r = {r:.2f} (p = {pr:.2f}); Spearman rho = {rs:.2f} (p = {ps:.2f})')
r2, pr2 = stats.pearsonr(m.realism, m.trust_total)
say(f'realism vs trust_total: Pearson r = {r2:.2f} (p = {pr2:.2f})')
m['pnum'] = m.pid.str[1:].astype(int)
r3, pr3 = stats.pearsonr(m.pnum, m.trust_total)
say(f'session order (participant number) vs trust_total: r = {r3:.2f} (p = {pr3:.2f})')
dur = m.duration_s / 60
say(f'duration (min): median {dur.median():.1f}, quartiles {dur.quantile(.25):.1f} to {dur.quantile(.75):.1f}, min {dur.min():.1f}, max {dur.max():.1f}; planned 13')
say('duration median by condition: ' + '; '.join(f'{c} {(m[m.condition==c].duration_s/60).median():.1f}' for c in 'ABCD'))
r4, pr4 = stats.pearsonr(dur, m.trust_total)
say(f'duration vs trust_total: r = {r4:.2f} (p = {pr4:.2f})')
# reply words
m['rw'] = m.reply_words.str.split('|').apply(lambda v: sum(int(x) for x in v))
say('total words of the three replies by condition (mean): ' + '; '.join(f'{c} {m[m.condition==c].rw.mean():.1f}' for c in 'ABCD'))
# scenarios seen: per scenario mean trust within accuracy
sc = []
for _, r_ in m.iterrows():
    for s in r_.scenario_ids.split('|'):
        sc.append(dict(scenario=s, accuracy=r_.accuracy, trust=r_.trust_total))
sc = pd.DataFrame(sc).groupby(['scenario', 'accuracy']).trust.agg(['count', 'mean']).round(2).reset_index()
sc.to_csv(f'{OUT}/table_by_scenario.csv', index=False)
sg = sc.pivot(index='scenario', columns='accuracy', values='mean'); sg['gap_correct_minus_incorrect'] = sg.correct - sg.incorrect; sg.round(2).to_csv(f'{OUT}/table_scenario_gap.csv'); say('gap per scenario (correct minus incorrect):'); say(sg.round(2).to_string())
say('mean trust by scenario seen (a session contributes to the 3 scenarios it saw) saved to table_by_scenario.csv')
say(sc.to_string(index=False))

# ---------------- 9. items & reliability --------------------------------------------------------
say('\n== 9. Items and reliability ==')
it = m[pts]
k = 10
alpha = k / (k - 1) * (1 - it.var(ddof=1).sum() / it.sum(axis=1).var(ddof=1))
say(f'Cronbach alpha (10 scores, 43 sessions) = {alpha:.2f}')
rows = []
for i in range(1, 11):
    col = f'pts{i}'
    a_, b_ = m[m.accuracy == 'correct'][col], m[m.accuracy == 'incorrect'][col]
    rest = it.drop(columns=col).sum(axis=1)
    rit = stats.pearsonr(m[col], rest)[0]
    rows.append(dict(question=i, mean_correct=a_.mean(), mean_incorrect=b_.mean(), difference=a_.mean() - b_.mean(),
                     median_correct=a_.median(), median_incorrect=b_.median(), item_rest_r=rit))
TI = pd.DataFrame(rows).round(2)
TI.to_csv(f'{OUT}/table_items.csv', index=False)
say(TI.to_string(index=False))
a_drop = {i: (k - 1) / (k - 2) * (1 - it.drop(columns=f'pts{i}').var(ddof=1).sum() / it.drop(columns=f'pts{i}').sum(axis=1).var(ddof=1)) for i in range(1, 11)}
say('alpha if question dropped: ' + ', '.join(f'Q{i} {v:.2f}' for i, v in a_drop.items()))
# trust without weakest items
weak = [1, 2, 4]
keep = [f'pts{i}' for i in range(1, 11) if i not in weak]
tt = m[keep].sum(axis=1)
dw = welch(tt[m.accuracy == 'correct'], tt[m.accuracy == 'incorrect'])
say(f'trust total without Q1, Q2, Q4 (max 28): correct-incorrect = {dw[0]:.2f} (CI {dw[1]:.2f} to {dw[2]:.2f}); as share of 28 = {dw[0]/28*100:.1f} percent, main score share of 40 = {d/40*100:.1f} percent')

# ---------------- 10. ER numbers kept in one place ----------------------------------
say('\n== 10. Extra numbers ==')
say(f'trust_band counts overall (main): {m.trust_band.value_counts().to_dict()}')
say(f'correct vs incorrect: share in trust band: correct {int((cor>=27).sum())} of {len(cor)}; incorrect {int((inc>=27).sum())} of {len(inc)}')
say(f'correct vs incorrect: share in distrust band (0-13): correct {int((cor<=13).sum())} of {len(cor)}; incorrect {int((inc<=13).sum())} of {len(inc)}')
say(f'incorrect-reply totals between 10 and 20: {int(((inc>=10)&(inc<=20)).sum())} of {len(inc)}; correct-reply totals between 24 and 36: {int(((cor>=24)&(cor<=36)).sum())} of {len(cor)}; lowest correct total {int(cor.min())}, highest incorrect total {int(inc.max())}')
ct = pd.crosstab(m.condition, m.check_correct)
say('check_correct counts by condition (columns = answer 0,1,2,3):')
say(ct.to_string())
say(f'lowest trust total: {m.trust_total.min()} ({m.loc[m.trust_total.idxmin(),"pid"]}); highest: {m.trust_total.max()} ({m.loc[m.trust_total.idxmax(),"pid"]})')
say(f'sessions in B in trust band: {", ".join(m[(m.condition=="B")&(m.trust_total>=27)].pid)}')
say(f'sessions in A below 27: {", ".join(m[(m.condition=="A")&(m.trust_total<27)].pid)}')


say('\n== 11. Derived numbers used in the text ==')
say(f'A mean minus 27 = {mu["A"]-27:.2f}; 28 minus A mean = {28-mu["A"]:.2f}; 26 minus C mean = {26-mu["C"]:.2f}')
say(f'margins over 8 points: unadjusted {d-8:.2f}; adjusted {-g1.params["acc_inc"]-8:.2f}; all 48 rows {d3-8:.2f}; matching-check subset {dd2-8:.2f}')
say(f'difference of check_correct means (correct minus incorrect) = {m[m.accuracy=="correct"].check_correct.mean()-m[m.accuracy=="incorrect"].check_correct.mean():.2f}; A minus B = {mu["A"]-mu["B"]:.2f}')
say(f'bootstrap share of resamples with difference >= 8: {np.mean(np.array(bs) >= 8)*100:.1f} percent')
say(f'adjusted interaction (drop complex minus drop simple, with prior use and realism) = {-f2.params["acc_inc:cx"]:.2f}, CI {-ci2.loc["acc_inc:cx"][1]:.2f} to {-ci2.loc["acc_inc:cx"][0]:.2f}')
say('sessions per start date (UTC), all 48 rows: ' + str(raw.started_at.str[:10].value_counts().sort_index().to_dict()))
say(f'trust answers: 10 x {len(m)} = {10*len(m)}')

open(f'{OUT}/a4_numbers.txt', 'w', encoding='utf-8').write('\n'.join(lines))

# ============================== FIGURES =========================================================
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 130})
rng = np.random.default_rng(7)

# G1 distribution by condition (box + points)
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for i, c in enumerate('ABCD'):
    x = m[m.condition == c].trust_total
    ax.boxplot(x, positions=[i], widths=.5, showfliers=False, medianprops=dict(color='black', lw=1.8),
               boxprops=dict(color=COL[c]), whiskerprops=dict(color=COL[c]), capprops=dict(color=COL[c]))
    ax.scatter(i + rng.uniform(-.14, .14, len(x)), x, s=22, color=COL[c], alpha=.75, zorder=3)
ax.axhspan(27, 40, color=BLUE, alpha=.06); ax.axhspan(0, 13.5, color=ORANGE, alpha=.06)
ax.axhline(27, color=GREY, ls=':', lw=1); ax.axhline(13.5, color=GREY, ls=':', lw=1)
ax.text(3.45, 33.5, 'trust band\n27-40', ha='right', fontsize=8, color=GREY)
ax.text(-.45, 5, 'distrust band 0-13', ha='left', fontsize=8, color=GREY)
ax.set_xticks(range(4)); ax.set_xticklabels([f'{LAB[c]}\n(n = {int((m.condition==c).sum())})' for c in 'ABCD'], fontsize=8.5)
ax.set_ylabel('Trust total (0-40 points)'); ax.set_ylim(0, 40)
ax.set_title('Trust total in each condition (box = quartiles, bar = median, dots = sessions)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{OUT}/g1_distribution_by_condition.png'); plt.close(fig)

# G2 interaction plot with CI
fig, ax = plt.subplots(figsize=(5.6, 4.2))
for acc, col, mk in [('correct', BLUE, 'o'), ('incorrect', ORANGE, 's')]:
    ys, es = [], []
    for dif in ['simple', 'complex']:
        x = m[(m.difficulty == dif) & (m.accuracy == acc)].trust_total
        ys.append(x.mean()); es.append(stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    ax.errorbar([0, 1], ys, yerr=es, color=col, marker=mk, capsize=4, lw=1.8, label=f'{acc} replies')
    for xx, yy in zip([0, 1], ys): ax.text(xx + .04, yy + 1.2, f'{yy:.1f}', color=col, fontsize=8.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(['simple questions', 'complex questions']); ax.set_xlim(-.3, 1.3)
ax.set_ylabel('Mean trust total (points), 95% CI'); ax.set_ylim(5, 35); ax.legend(frameon=False, loc='lower left')
ax.set_title('Mean trust: difficulty x accuracy', fontsize=10)
fig.tight_layout(); fig.savefig(f'{OUT}/g2_interaction.png'); plt.close(fig)

# G3 differences vs baseline with CI, threshold line
fig, ax = plt.subplots(figsize=(6.4, 3.4))
labs = ['B vs A\n(incorrect, simple)', 'C vs A\n(complex, correct)', 'D vs A\n(incorrect, complex)']
for i, (c, r_) in enumerate(zip('BCD', TD.itertuples())):
    ax.errorbar(r_.mean_diff_vs_A, i, xerr=[[r_.mean_diff_vs_A - r_.ci_low], [r_.ci_high - r_.mean_diff_vs_A]], fmt='o', color=COL[c], capsize=4)
    ax.text(r_.ci_high + .5, i, f'{r_.mean_diff_vs_A:.1f} [{r_.ci_low:.1f}, {r_.ci_high:.1f}]', va='center', fontsize=8.5)
ax.axvline(0, color='black', lw=1); ax.axvline(-8, color=GREY, ls='--', lw=1)
ax.text(-8, 2.55, 'declared minimum\n-8 points', ha='center', fontsize=8, color=GREY)
ax.set_yticks(range(3)); ax.set_yticklabels(labs, fontsize=8.5); ax.set_xlim(-24, 8); ax.set_ylim(-.5, 3.1); ax.invert_yaxis()
ax.set_xlabel('Difference in mean trust total from baseline A (points), 95% CI')
fig.tight_layout(); fig.savefig(f'{OUT}/g3_difference_from_baseline.png'); plt.close(fig)

# G4 histograms correct vs incorrect
fig, ax = plt.subplots(figsize=(6.8, 3.8))
bins = np.arange(0, 42, 4)
ax.hist(cor, bins=bins, color=BLUE, alpha=.65, label=f'correct replies (n = {len(cor)})')
ax.hist(inc, bins=bins, color=ORANGE, alpha=.65, label=f'incorrect replies (n = {len(inc)})')
ax.axvline(cor.mean(), color=BLUE, lw=1.5, ls='--'); ax.axvline(inc.mean(), color=ORANGE, lw=1.5, ls='--')
ax.axvline(27, color=GREY, ls=':'); ax.axvline(13.5, color=GREY, ls=':')
ax.set_xlabel('Trust total (points, bins of 4)'); ax.set_ylabel('Number of sessions'); ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
ax.set_title('Distribution of trust totals (dashed = means, dotted = band limits)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{OUT}/g4_histogram_accuracy.png'); plt.close(fig)

# G5 item level difference
fig, ax = plt.subplots(figsize=(6.8, 3.8))
cols = [ORANGE if v < 0.6 else BLUE for v in TI.difference]
ax.bar(TI.question.astype(str), TI.difference, color=cols)
for x_, v in zip(TI.question.astype(str), TI.difference): ax.text(x_, v + .03, f'{v:.2f}', ha='center', fontsize=8)
ax.set_xlabel('Trust question number'); ax.set_ylabel('Mean points: correct minus incorrect (0-4 scale)')
ax.set_title('Question by question: correct minus incorrect (orange = below 0.6 points)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{OUT}/g5_item_differences.png'); plt.close(fig)

# G6 check_correct distribution by condition (stacked bar)
fig, ax = plt.subplots(figsize=(6.8, 3.8))
bottom = np.zeros(4)
shades = ['#d95f02', '#f1a340', '#9ecae1', '#1f77b4']
for v, sh in zip(range(4), shades):
    h = np.array([(m[m.condition == c].check_correct == v).sum() for c in 'ABCD'])
    ax.bar(list('ABCD'), h, bottom=bottom, color=sh, label=f'{v} of 3 replies judged correct')
    for i_, (hh, bb) in enumerate(zip(h, bottom)):
        if hh: ax.text(i_, bb + hh / 2, str(hh), ha='center', va='center', fontsize=8, color='white' if v in (0, 3) else 'black')
    bottom += h
ax.set_ylabel('Number of sessions'); ax.set_xlabel('Condition'); ax.legend(frameon=False, fontsize=7.5, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
ax.set_title('Check question: how many of the 3 replies were correct, in your view?', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{OUT}/g6_check_correct.png'); plt.close(fig)

# G7 trust vs realism / prior use scatter
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
for ax, col, lab in [(axs[0], 'prior_use', 'Prior chatbot use (1-5)'), (axs[1], 'realism', 'Realism of the situations (1-7)')]:
    for acc, cc_ in [('correct', BLUE), ('incorrect', ORANGE)]:
        s = m[m.accuracy == acc]
        ax.scatter(s[col] + rng.uniform(-.12, .12, len(s)), s.trust_total, color=cc_, alpha=.7, s=22, label=f'{acc} replies')
    ax.set_xlabel(lab)
axs[0].set_ylabel('Trust total (points)'); axs[0].legend(frameon=False, fontsize=8)
fig.suptitle(f'Trust against the two possible confounders (r = {r:.2f} and {r2:.2f})', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{OUT}/g7_confounders.png'); plt.close(fig)

# G8 duration by condition
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for i, c in enumerate('ABCD'):
    x = m[m.condition == c].duration_s / 60
    ax.scatter(i + rng.uniform(-.14, .14, len(x)), x, color=COL[c], s=22, alpha=.75)
    ax.hlines(x.median(), i - .25, i + .25, color='black', lw=1.8)
ax.axhline(13, color=GREY, ls='--', lw=1); ax.text(3.4, 13.4, 'planned 13 min', ha='right', fontsize=8, color=GREY)
ax.set_xticks(range(4)); ax.set_xticklabels(list('ABCD')); ax.set_xlabel('Condition'); ax.set_ylabel('Session length (minutes)')
ax.set_title('Session length by condition (bar = median)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{OUT}/g8_duration.png'); plt.close(fig)

# G9 difficulty rating (check)
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for i, dif in enumerate(['simple', 'complex']):
    x = m[m.difficulty == dif].check_difficulty
    ax.scatter(i + rng.uniform(-.14, .14, len(x)), x, color=GREY, alpha=.6, s=24)
    ax.scatter(i, x.mean(), marker='D', color='black', s=50, zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels(['simple', 'complex']); ax.set_ylabel('Difficulty rating (1-7)'); ax.set_ylim(.5, 7.5)
ax.set_title('Participants rate complex situations as harder (diamond = mean)', fontsize=9.5)
fig.tight_layout(); fig.savefig(f'{OUT}/g9_difficulty_rating.png'); plt.close(fig)
print('done')
