"""Lean learning-curve run (one dataset per call). Blocked 5-fold; training subsets drawn at random
from each fold's training portion; scored on the untouched held-out block."""
import sys, time, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from common import *
quiet()
dname = sys.argv[1]
df, X, y = load_aemo() if dname == 'aemo' else load_kaggle()
FRACS = [0.1, 0.25, 0.5, 0.75, 1.0]
REPS = {'kNN': 5, 'MLP': 1}
rows, t0 = [], time.time()
for model in ['kNN', 'MLP']:
    for fold, (tr, te) in enumerate(KFold(5, shuffle=False).split(X)):
        for f in FRACS:
            for rep in range(REPS[model]):
                rng = np.random.RandomState(1000 * rep + fold)
                sub = rng.choice(tr, size=max(int(round(f * len(tr))), 10), replace=False)
                m = make_model(model, seed=rep).fit(X.iloc[sub], y.iloc[sub])
                rows.append(dict(Dataset=dname, Model=model, frac=f, n_train=len(sub), fold=fold, rep=rep,
                                 test_RMSE=metrics(y.iloc[te], m.predict(X.iloc[te]))['RMSE'],
                                 train_RMSE=metrics(y.iloc[sub], m.predict(X.iloc[sub]))['RMSE']))
    print(model, 'done', round(time.time() - t0), 's', flush=True)
pd.DataFrame(rows).to_csv(f'out/lc_{dname}.csv', index=False)
