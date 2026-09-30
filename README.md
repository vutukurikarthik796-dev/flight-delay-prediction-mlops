# Flight Delay Prediction — MLOps Project

This project restructures the original Flight Delay Prediction work into a Lab 3–6 MLOps workflow similar to the churn-prediction project.

## Structure
- `data/` — original flight, airline and airport CSV files
- `models/` — original model plus generated models
- `artifacts/` — preprocessing and model metadata
- `outputs/` — evaluation results
- `notebooks/` — original implementation notebook
- `src/` — validation, preprocessing and training code
- `pipelines/` — Lab 3, Lab 4, Lab 5 and Lab 6 runners

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run in order
```powershell
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
python pipelines/run_lab5_pipeline.py
python pipelines/run_lab6_registry.py
```

## MLflow
After Lab 4 or Lab 6:
```powershell
mlflow ui
```
Open `http://127.0.0.1:5000`.

The training scripts use the first 100,000 usable rows by default so the pipeline is practical on a student laptop. Change `NROWS` in `src/train_models.py` if you want to train on more data.
