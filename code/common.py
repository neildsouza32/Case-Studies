"""Shared loaders + models for Task 2. Preprocessing mirrors Task 1 exactly
(electricity.py / signals.py) so results are comparable with Table 1."""
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

KAGGLE_FEATURES = ['RRP', 'min_temperature', 'max_temperature', 'solar_exposure',
                   'rainfall', 'school_day', 'holiday', 'month', 'dayofweek']
AEMO_FEATURES = ['hour', 'dayofweek', 'QLD1', 'SA1', 'TAS1', 'VIC1']


def load_kaggle(path="electricity_demand_victoria.csv"):
    df = pd.read_csv(path).dropna()          # Task 1 dropped 4 rows with NaNs
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['school_day'] = df['school_day'].map({'Y': 1, 'N': 0})
    df['holiday'] = df['holiday'].map({'Y': 1, 'N': 0})
    df = df.sort_values('date').reset_index(drop=True)
    return df, df[KAGGLE_FEATURES], df['demand']


def load_aemo(path="signals_regional_load.csv"):
    df = pd.read_csv(path)
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'])
    df['hour'] = df['SETTLEMENTDATE'].dt.hour
    df['dayofweek'] = df['SETTLEMENTDATE'].dt.dayofweek
    df = df.sort_values('SETTLEMENTDATE').reset_index(drop=True)
    return df, df[AEMO_FEATURES], df['NSW1']


def make_model(name, seed=42):
    """Scaler lives INSIDE the pipeline so it is re-fitted on each training fold
    (no scaling leakage during cross-validation)."""
    if name == 'kNN':
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    if name == 'MLP':
        return make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000,
                                          random_state=seed))
    raise ValueError(name)


def metrics(y_true, y_pred):
    return dict(MAE=mean_absolute_error(y_true, y_pred),
                RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
                R2=r2_score(y_true, y_pred))


def quiet():
    warnings.filterwarnings("ignore")
