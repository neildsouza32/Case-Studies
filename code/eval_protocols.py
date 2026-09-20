"""Part 2 (items 1 & 2): does the evaluation give an unbiased estimate, and how does
performance change with training-set size?  Writes CSVs + figures to ./out"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from common import *

quiet()
os.makedirs("out", exist_ok=True)
DATA = {'AEMO regional load': load_aemo(), 'Kaggle Victoria demand': load_kaggle()}
MODELS = ['kNN', 'MLP']


def fit_eval(model, X, y, tr, te, seed=42):
    m = make_model(model, seed)
    m.fit(X.iloc[tr], y.iloc[tr])
    return metrics(y.iloc[te], m.predict(X.iloc[te]))


# ------------------------------------------------------------------ protocols
def protocol_rows(dname, model, X, y):
    n = len(X)
    idx = np.arange(n)
    rows = []

    # (a) the single random 80/20 split used in Task 1
    tr, te = train_test_split(idx, test_size=0.2, random_state=42)
    r = fit_eval(model, X, y, tr, te)
    rows.append(('Single random 80/20 (Task 1)', r['RMSE'], np.nan, r['MAE'], r['R2']))

    # (b) same split idea repeated with 30 different seeds -> how lucky was seed 42?
    def one(seed):
        tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
        return fit_eval(model, X, y, tr, te, seed)
    NSEED = 30 if model == 'kNN' else 10
    rs = [one(s) for s in range(NSEED)]
    rows.append((f'Repeated random 80/20 ({NSEED} seeds)', np.mean([r['RMSE'] for r in rs]),
                 np.std([r['RMSE'] for r in rs]), np.mean([r['MAE'] for r in rs]),
                 np.mean([r['R2'] for r in rs])))

    def cv_rows(label, splitter):
        rs = [fit_eval(model, X, y, tr, te) for tr, te in splitter.split(idx)]
        rows.append((label, np.mean([r['RMSE'] for r in rs]), np.std([r['RMSE'] for r in rs]),
                     np.mean([r['MAE'] for r in rs]), np.mean([r['R2'] for r in rs])))

    # (c) shuffled 5-fold CV  (i.i.d. assumption)
    cv_rows('Shuffled 5-fold CV', KFold(5, shuffle=True, random_state=42))
    # (d) blocked 5-fold CV: contiguous blocks, no shuffling -> neighbours not split across train/test
    cv_rows('Blocked 5-fold CV', KFold(5, shuffle=False))
    # (e) forward-chaining: train on the past only, test on the next block
    cv_rows('Forward-chaining (5 splits)', TimeSeriesSplit(5))
    return rows


res = []
for dname, (df, X, y) in DATA.items():
    for model in MODELS:
        for lab, rmse, sd, mae, r2 in protocol_rows(dname, model, X, y):
            res.append(dict(Dataset=dname, Model=model, Protocol=lab, RMSE=rmse, RMSE_sd=sd, MAE=mae, R2=r2))
        print("protocols done:", dname, model, flush=True)
prot = pd.DataFrame(res)
prot.to_csv("out/protocol_comparison.csv", index=False)
print(prot.round(3).to_string())

# ------------------------------------------------------------------ learning curves
FRACS = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0])
REPEATS = {'kNN': 5, 'MLP': 2}


def lc_task(dname, model, fold, tr, te, frac, rep):
    df, X, y = DATA[dname]
    rng = np.random.RandomState(1000 * rep + fold)
    k = max(int(round(frac * len(tr))), 10)
    sub = rng.choice(tr, size=k, replace=False)
    m = make_model(model, seed=rep)
    m.fit(X.iloc[sub], y.iloc[sub])
    te_r = metrics(y.iloc[te], m.predict(X.iloc[te]))
    tr_r = metrics(y.iloc[sub], m.predict(X.iloc[sub]))
    return dict(Dataset=dname, Model=model, frac=frac, n_train=k, fold=fold, rep=rep,
                test_RMSE=te_r['RMSE'], train_RMSE=tr_r['RMSE'], test_R2=te_r['R2'])


tasks = []
for dname, (df, X, y) in DATA.items():
    folds = list(KFold(5, shuffle=False).split(np.arange(len(X))))
    for model in MODELS:
        for fold, (tr, te) in enumerate(folds):
            for frac in FRACS:
                for rep in range(REPEATS[model]):
                    tasks.append((dname, model, fold, tr, te, frac, rep))
print("learning-curve fits:", len(tasks), flush=True)
lc = pd.DataFrame([lc_task(*t) for t in tasks])
lc.to_csv("out/learning_curve_raw.csv", index=False)
agg = (lc.groupby(['Dataset', 'Model', 'frac'])
         .agg(n_train=('n_train', 'mean'), test_RMSE=('test_RMSE', 'mean'), test_sd=('test_RMSE', 'std'),
              train_RMSE=('train_RMSE', 'mean'), test_R2=('test_R2', 'mean')).reset_index())
agg.to_csv("out/learning_curve.csv", index=False)
print(agg.round(2).to_string())

# ------------------------------------------------------------------ figures
plt.rcParams.update({'font.size': 9})
# Fig 1 - protocol comparison
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
labels = ['Single random\n80/20 (Task 1)', 'Repeated random\n80/20', 'Shuffled\n5-fold', 'Blocked\n5-fold', 'Forward-\nchaining']
for ax, dname in zip(axes, DATA):
    sub = prot[prot.Dataset == dname]
    w = 0.38
    for j, model in enumerate(MODELS):
        s = sub[sub.Model == model].reset_index(drop=True)
        ax.bar(np.arange(5) + (j - .5) * w, s.RMSE, w, yerr=s.RMSE_sd.fillna(0), capsize=2, label=model)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_title(dname, fontsize=9); ax.set_ylabel('RMSE'); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("out/fig_protocols.pdf"); plt.savefig("out/fig_protocols.png", dpi=200); plt.close()

# Fig 2 - learning curves
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
colors = {'kNN': 'tab:blue', 'MLP': 'tab:orange'}
for ax, dname in zip(axes, DATA):
    for model in MODELS:
        s = agg[(agg.Dataset == dname) & (agg.Model == model)]
        ax.plot(s.n_train, s.test_RMSE, '-o', ms=3, color=colors[model], label=f'{model} (held-out block)')
        ax.fill_between(s.n_train, s.test_RMSE - s.test_sd, s.test_RMSE + s.test_sd, color=colors[model], alpha=.15)
        ax.plot(s.n_train, s.train_RMSE, '--', color=colors[model], label=f'{model} (training)')
    ax.set_title(dname, fontsize=9); ax.set_xlabel('Training-set size (rows)'); ax.set_ylabel('RMSE')
    ax.legend(fontsize=6.5)
plt.tight_layout(); plt.savefig("out/fig_learning_curves.pdf"); plt.savefig("out/fig_learning_curves.png", dpi=200); plt.close()
print("done")
