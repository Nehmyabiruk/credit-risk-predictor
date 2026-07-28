# Loan Credit Risk – Production ML Pipeline

Professional restructuring of the original EDA + modelling notebooks
(`eda.ipynb` / `edaa.ipynb`) into a clean, reproducible project used by
real ML / AI engineering teams.

## Project layout

```
loan-credit-risk/
├── data/
│   ├── raw/                  # place loan_data.csv here
│   └── processed/            # cleaned artefacts (written by preprocess)
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # cleaning + feature engineering
│   ├── train.py              # split → encode → scale → tune → save
│   ├── evaluate.py           # metrics, ROC/PR plots, confusion matrix
│   ├── predict.py            # inference helper (CLI + Python API)
│   └── utils.py              # shared helpers (config, logging, paths)
├── models/                   # .pkl artefacts after training
├── reports/                  # metrics.json + plots from evaluate.py
├── configs/
│   └── config.yaml           # hyper-parameters & column lists
├── notebooks/                # original exploratory notebooks
├── tests/                    # smoke tests (pytest)
├── Makefile                  # one-command workflows
├── requirements.txt
├── .gitignore
└── README.md
```

## Quick start

```bash
# 1. Install
pip install -r requirements.txt
# or: make install

# 2. Put the original CSV in the expected location
cp /path/to/loan_data.csv data/raw/

# 3. Full pipeline
make all
# equivalent to:
#   python -m src.preprocess
#   python -m src.train
#   python -m src.evaluate

# 4. Score new applicants
python -m src.predict --csv new_applicants.csv --out scored.csv
```

## Scripts (what each does)

| Script | Responsibility |
|--------|----------------|
| **`preprocess.py`** | Age clip ≤ 80, drop extreme incomes (> $3 M), log-income + 99.9 % cap, engineer `debt_income` / `age_ratio` / `loan_history`, write clean CSVs |
| **`train.py`** | Stratified 80/20 split → `ColumnTransformer` (OneHot + StandardScaler) → `RandomizedSearchCV` on XGBoost (20 iters, 5-fold ROC-AUC) → save full pipeline + model + scaler |
| **`evaluate.py`** | Load saved pipeline, score hold-out (or external CSV), print classification report, ROC-AUC / PR-AUC, write `reports/metrics.json` + confusion / ROC / PR plots |
| **`predict.py`** | Load full pipeline and return default probability for any CSV with the original feature columns |
| **`utils.py`** | Shared logging, config loader, path helpers |

## Evaluation outputs (`reports/`)

After `python -m src.evaluate` (or `make evaluate`):

- `metrics.json` – accuracy, precision/recall/F1 per class, ROC-AUC, PR-AUC, threshold
- `confusion_matrix.png`
- `roc_curve.png`
- `pr_curve.png`

You can also evaluate an external labelled CSV and change the decision threshold:

```bash
python -m src.evaluate --test-csv my_test.csv --threshold 0.4
```

## Model performance (from original notebook)

| Metric     | Class 0 (no default) | Class 1 (default) | Overall |
|------------|----------------------|-------------------|---------|
| Precision  | 0.95                 | 0.89              | –       |
| Recall     | 0.97                 | 0.81              | –       |
| F1         | 0.96                 | 0.85              | –       |
| Accuracy   | –                    | –                 | 0.94    |
| CV ROC-AUC | –                    | –                 | 0.979   |

## Design principles (how real ML teams structure this)

1. **Single responsibility** – preprocess / train / evaluate / predict are independent entry points.
2. **Reproducible** – fixed random seeds, centralised paths, no hidden notebook state.
3. **Full sklearn Pipeline** is the deployment artefact – inference never re-implements encoding or scaling.
4. **Config-driven** – column lists and search space live in `configs/config.yaml`.
5. **Observable** – dedicated evaluate stage writes metrics + plots to `reports/`.
6. **Testable** – smoke tests under `tests/` (run with `pytest`).
7. **Makefile** – one-command workflows for the whole team.
8. **Notebooks stay exploratory** – production logic lives in `src/`.

## Makefile targets

```
make install      # pip install -r requirements.txt
make preprocess   # clean + feature engineer
make train        # train + hyperparameter search
make evaluate     # metrics + plots on hold-out
make all          # preprocess → train → evaluate
make clean        # remove processed data, models, reports
```

## Next steps (common upgrades)

- Wire `configs/config.yaml` into every script (currently values are also hard-coded for clarity)
- Add `pyproject.toml` so the package is installable (`pip install -e .`)
- Track experiments with MLflow or Weights & Biases
- Serve the model with FastAPI / BentoML
- Add data-validation (Great Expectations / pandera) before training
