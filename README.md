# Data Science Career Portfolio — Analyst, Market Modelling (AEMO)

This repository contains the code, datasets, and analysis supporting my assignment on the *Analyst - Market Modelling* role at the Australian Energy Market Operator (AEMO).

## Overview

Two machine learning models — **k-Nearest Neighbours (kNN) Regression** and a **Neural Network (MLP Regressor)** — were applied separately to two datasets related to electricity demand forecasting in the Australian National Electricity Market (NEM), to generate insights relevant to the target job role.



## Datasets

| Dataset | Source | Description |
|---|---|---|
| AEMO Regional Load | [egrimod-nem-dataset (GitHub)](https://github.com/akxen/egrimod-nem-dataset), originally AEMO | 30-minute interval electricity demand across 5 NEM regions (NSW, QLD, SA, TAS, VIC), June 2017 |
| Victoria Electricity Demand | [Kaggle](https://www.kaggle.com/datasets/aramacus/electricity-demand-in-victoria-australia) | Daily electricity demand, price (RRP), and weather data for Victoria, 2015–2020 |

## Models Used

1. **k-Nearest Neighbours (kNN) Regressor** — `n_neighbors=5`
2. **Multi-Layer Perceptron (MLP) Neural Network Regressor** — hidden layers `(64, 32)`

Both models were trained and evaluated independently on each dataset (not merged), due to overlapping source data between the two datasets.

## Evaluation Metrics

- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of Determination)

RMSE was prioritised as the primary metric, given the importance of penalising large forecasting errors in an energy market reliability context.

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn
```

### 2. Place datasets in the `data/` folder

Download both datasets from the links above and save them as:
- `data/signals_regional_load.csv`
- `data/electricity_demand_victoria.csv`

### 3. Run the scripts

```bash
python notebooks/run_aemo_models.py
python notebooks/run_kaggle_models.py
```

Each script prints:
- Dataset shape
- MAE, RMSE, and R² for both models
- Feature importance (via permutation importance)

## Results Summary

| Dataset | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| AEMO Regional Load | kNN | 154.74 | 201.56 | 0.969 |
| AEMO Regional Load | MLP Neural Net | 200.12 | 255.17 | 0.950 |
| Kaggle Victoria Demand | kNN | 5072.40 | 6771.62 | 0.757 |
| Kaggle Victoria Demand | MLP Neural Net | 4672.11 | 6132.55 | 0.801 |

## Context

This analysis was completed as part of a data science career portfolio assignment, tied to the job advertisement for [Analyst - Market Modelling at AEMO](https://www.linkedin.com/jobs/view/4447106001/).

## Acknowledgements

- AEMO for publishing the original NEM market data.
- [akxen/egrimod-nem-dataset](https://github.com/akxen/egrimod-nem-dataset) for the packaged regional load dataset.
- Kaggle contributor *aramacus* for the Victoria electricity demand dataset.

