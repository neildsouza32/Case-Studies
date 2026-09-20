import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = "/home/claude/res/"
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
C = {'kNN': '#1f77b4', 'MLP': '#ff7f0e'}
# ---------- Fig 1: protocols
p = pd.read_csv(R+'protocol_comparison.csv')
labels = ['Single random\n80/20 (Task 1)', 'Repeated random\n80/20', 'Shuffled\n5-fold', 'Blocked\n5-fold', 'Forward-\nchaining']
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, d in zip(axes, p.Dataset.unique()):
    for j, m in enumerate(['kNN', 'MLP']):
        s = p[(p.Dataset == d) & (p.Model == m)].reset_index(drop=True)
        ax.bar(np.arange(5) + (j-.5)*.38, s.RMSE, .38, yerr=s.RMSE_sd.fillna(0), capsize=2, color=C[m], label=m)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=6); ax.set_ylabel('RMSE' + (' (MW)' if 'AEMO' in d else ' (MWh/day)'))
    ax.set_title(d, fontsize=8.5); ax.legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig('fig_protocols.pdf'); plt.close()
# ---------- Fig 2: learning curves
lc = pd.concat([pd.read_csv(R+'lc_aemo.csv'), pd.read_csv(R+'lc_kaggle.csv')])
lc['Dataset'] = lc.Dataset.map({'aemo': 'AEMO regional load', 'kaggle': 'Kaggle Victoria demand'})
g = lc.groupby(['Dataset', 'Model', 'frac']).agg(n=('n_train', 'mean'), te=('test_RMSE', 'mean'), sd=('test_RMSE', 'std'), tr=('train_RMSE', 'mean')).reset_index()
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, d in zip(axes, ['AEMO regional load', 'Kaggle Victoria demand']):
    for m in ['kNN', 'MLP']:
        s = g[(g.Dataset == d) & (g.Model == m)]
        ax.plot(s.n, s.te, '-o', ms=3, color=C[m], label=f'{m}: held-out block')
        ax.fill_between(s.n, s.te - s.sd, s.te + s.sd, color=C[m], alpha=.15)
        ax.plot(s.n, s.tr, '--', color=C[m], label=f'{m}: training')
    ax.set_xlabel('Training-set size (rows)'); ax.set_ylabel('RMSE'); ax.set_title(d, fontsize=8.5)
    if 'Kaggle' in d:
        ax.set_ylim(4000, 16000); ax.annotate('MLP at 168 rows:\n49,447 (off scale)', xy=(168, 15800), xytext=(420, 14200), fontsize=6.5, arrowprops=dict(arrowstyle='->', lw=.6))
    else:
        ax.set_ylim(100, 700)
    ax.legend(fontsize=6, frameon=False)
plt.tight_layout(); plt.savefig('fig_learning_curves.pdf'); plt.close()
# ---------- Fig 3: subgroup bias
f = pd.read_csv(R+'fairness_by_group.csv'); f = f[f.n >= 20]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'width_ratios': [1, 1]})
k = f[(f.Dataset.str.startswith('Kaggle')) & f.Attribute.isin(['Season', 'Max temperature', 'School day', 'Day type'])]
order = k[k.Model == 'kNN'].sort_values(['Attribute', 'Group'])[['Attribute', 'Group']].values.tolist()
y = np.arange(len(order)); ax = axes[0]
for j, m in enumerate(['kNN', 'MLP']):
    v = [k[(k.Model == m) & (k.Attribute == a) & (k.Group == gr)]['MAPE_%'].iloc[0] for a, gr in order]
    ax.barh(y + (j-.5)*.38, v, .38, color=C[m], label=m)
ax.set_yticks(y); ax.set_yticklabels([gr for a, gr in order], fontsize=6.5); ax.invert_yaxis()
ax.set_xlabel('MAPE (%)'); ax.set_title('(a) Kaggle: error by subgroup', fontsize=8.5); ax.legend(fontsize=7, frameon=False)
a_ = f[f.Dataset.str.startswith('AEMO')]
order = a_[a_.Model == 'kNN'].sort_values(['Attribute', 'Group'])[['Attribute', 'Group']].values.tolist()
y = np.arange(len(order)); ax = axes[1]
for j, m in enumerate(['kNN', 'MLP']):
    v = [a_[(a_.Model == m) & (a_.Attribute == a) & (a_.Group == gr)]['MeanSignedErr'].iloc[0] for a, gr in order]
    ax.barh(y + (j-.5)*.38, v, .38, color=C[m])
ax.axvline(0, color='k', lw=.6); ax.set_yticks(y); ax.set_yticklabels([gr for a, gr in order], fontsize=6.5); ax.invert_yaxis()
ax.set_xlabel('Mean signed error (MW); <0 = under-forecast'); ax.set_title('(b) AEMO: forecast bias by subgroup', fontsize=8.5)
plt.tight_layout(); plt.savefig('fig_fairness.pdf'); plt.savefig('fig_fairness_preview.png', dpi=110); plt.close()
print("ok")
