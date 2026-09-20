"""Part 2 (item 3): subgroup performance analysis with Fairlearn's MetricFrame.

Neither dataset contains personal/protected attributes, so 'sensitive features' here are
operationally meaningful subgroups (day type, holiday, season, heat, time of day, load level).
The fairness question is: is forecast reliability uneven across groups?

Predictions are OUT-OF-FOLD (blocked 5-fold CV) so every row is scored by a model that
did not train on it.

Backend: uses fairlearn.metrics.MetricFrame if installed, else an equivalent pandas groupby.
pip install fairlearn   # then re-run to obtain the Fairlearn-computed tables
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_predict
from common import *

quiet()
os.makedirs("out", exist_ok=True)

try:
    from fairlearn.metrics import MetricFrame
    import fairlearn
    BACKEND = f"Fairlearn {fairlearn.__version__} MetricFrame"
except ImportError:
    MetricFrame = None
    BACKEND = "pandas groupby (Fairlearn-equivalent; fairlearn not installed)"
print("Backend:", BACKEND)
open("out/fairness_backend.txt", "w").write(BACKEND)

# ---- subgroup metrics: every metric has signature f(y_true, y_pred) ----
def n_rows(yt, yp): return len(yt)
def mae(yt, yp): return float(np.mean(np.abs(yt - yp)))
def rmse(yt, yp): return float(np.sqrt(np.mean((yt - yp) ** 2)))
def mape(yt, yp): return float(np.mean(np.abs(yt - yp) / np.abs(yt)) * 100)
def signed_err(yt, yp): return float(np.mean(yp - yt))            # >0 over-forecast, <0 under-forecast
def under_share(yt, yp): return float(np.mean(yp < yt) * 100)     # % of rows under-forecast
METRICS = {'n': n_rows, 'MAE': mae, 'RMSE': rmse, 'MAPE_%': mape,
           'MeanSignedErr': signed_err, 'Under_%': under_share}


def by_group(y, pred, g):
    y = np.asarray(y, float); pred = np.asarray(pred, float); g = pd.Series(np.asarray(g))
    if MetricFrame is not None:
        mf = MetricFrame(metrics=METRICS, y_true=y, y_pred=pred, sensitive_features=g)
        return mf.by_group.copy(), pd.Series(mf.overall)
    rows = {}
    for lv in g.unique():
        m = (g == lv).to_numpy()
        rows[lv] = {k: f(y[m], pred[m]) for k, f in METRICS.items()}
    overall = pd.Series({k: f(y, pred) for k, f in METRICS.items()})
    return pd.DataFrame(rows).T, overall


# ---- group definitions ----
def kaggle_groups(df):
    season = df['date'].dt.month.map(lambda m: 'Summer (Dec-Feb)' if m in (12, 1, 2) else
                                     'Autumn (Mar-May)' if m in (3, 4, 5) else
                                     'Winter (Jun-Aug)' if m in (6, 7, 8) else 'Spring (Sep-Nov)')
    return {
        'Day type': np.where(df['dayofweek'] >= 5, 'Weekend', 'Weekday'),
        'Public holiday': np.where(df['holiday'] == 1, 'Holiday', 'Non-holiday'),
        'School day': np.where(df['school_day'] == 1, 'School day', 'Non-school day'),
        'Season': season.to_numpy(),
        'Max temperature': np.where(df['max_temperature'] >= 35, 'Extreme heat (>=35C)',
                                    np.where(df['max_temperature'] >= 25, 'Warm (25-35C)', 'Mild (<25C)')),
        'Price regime': np.where(df['RRP'] >= 300, 'Price spike (RRP>=300)', 'Normal (RRP<300)'),
        'Year': df['date'].dt.year.astype(str).to_numpy(),
    }


def aemo_groups(df):
    h = df['hour']
    band = np.select([h < 6, h < 10, h < 16, h < 21], ['Overnight (0-6)', 'Morning ramp (6-10)',
                                                          'Daytime (10-16)', 'Evening peak (16-21)'],
                     default='Late evening (21-24)')
    q90 = df['NSW1'].quantile(0.9)
    return {
        'Time of day': band,
        'Day type': np.where(df['dayofweek'] >= 5, 'Weekend', 'Weekday'),
        'Load level': np.where(df['NSW1'] >= q90, 'Top-10% load', 'Other 90%'),
    }


SETS = {'AEMO regional load': (load_aemo(), aemo_groups),
        'Kaggle Victoria demand': (load_kaggle(), kaggle_groups)}

all_rows, summ_rows = [], []
oof_store = {}
for dname, ((df, X, y), gfun) in SETS.items():
    groups = gfun(df)
    for model in ['kNN', 'MLP']:
        pred = cross_val_predict(make_model(model), X, y, cv=KFold(5, shuffle=False))
        oof_store[(dname, model)] = pred
        print("OOF predictions:", dname, model, flush=True)
        for attr, g in groups.items():
            tab, overall = by_group(y, pred, g)
            for lv, r in tab.iterrows():
                all_rows.append(dict(Dataset=dname, Model=model, Attribute=attr, Group=lv, **r.to_dict()))
            # 'between-groups' summary (same definition as MetricFrame.difference / .ratio)
            keep = tab[tab['n'] >= 20]                       # ignore tiny groups in gap summary
            for met in ['MAE', 'RMSE', 'MAPE_%']:
                summ_rows.append(dict(Dataset=dname, Model=model, Attribute=attr, Metric=met,
                                      Overall=overall[met], Best=keep[met].idxmin(), BestVal=keep[met].min(),
                                      Worst=keep[met].idxmax(), WorstVal=keep[met].max(),
                                      Difference=keep[met].max() - keep[met].min(),
                                      Ratio_worst_over_best=keep[met].max() / keep[met].min()))
full = pd.DataFrame(all_rows)
summ = pd.DataFrame(summ_rows)
full.to_csv("out/fairness_by_group.csv", index=False)
summ.to_csv("out/fairness_summary.csv", index=False)
pd.to_pickle({k: v for k, v in oof_store.items()}, "out/oof_predictions.pkl")

pd.set_option('display.width', 250)
print(full.round(2).to_string())
print(summ[summ.Metric == 'MAPE_%'].round(2).to_string())
