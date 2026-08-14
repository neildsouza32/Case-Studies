import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

# ---- Load data ----
df = pd.read_csv("electricity_demand_victoria.csv")  # adjust path
df = df.dropna()
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df['school_day'] = df['school_day'].map({'Y': 1, 'N': 0})
df['holiday'] = df['holiday'].map({'Y': 1, 'N': 0})

print("Shape:", df.shape)

# ---- Features & target ----
target = 'demand'
feature_cols = ['RRP', 'min_temperature', 'max_temperature', 'solar_exposure',
                 'rainfall', 'school_day', 'holiday', 'month', 'dayofweek']
X = df[feature_cols]
y = df[target]

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- Scale ----
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ---- Model 1: kNN ----
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_s, y_train)
pred_knn = knn.predict(X_test_s)

# ---- Model 2: MLP Neural Network ----
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=42)
mlp.fit(X_train_s, y_train)
pred_mlp = mlp.predict(X_test_s)

# ---- Evaluation ----
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}: MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}")
    return mae, rmse, r2

print("\n=== Kaggle Victoria Demand Dataset (target: demand) ===")
evaluate(y_test, pred_knn, "kNN")
evaluate(y_test, pred_mlp, "MLP Neural Net")

# ---- Feature importance ----
result = permutation_importance(mlp, X_test_s, y_test, n_repeats=10, random_state=42)
imp = pd.DataFrame({'feature': feature_cols, 'importance': result.importances_mean}).sort_values('importance', ascending=False)
print("\nFeature importance (MLP):\n", imp)