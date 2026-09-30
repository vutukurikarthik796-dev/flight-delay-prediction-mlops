import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, subprocess, sys
try:
 import mlflow
except ImportError:
 print('[ERROR] MLflow is not installed. Run: python -m pip install -r requirements.txt'); raise
print('[INFO] =========================================')
print('[INFO] Starting Lab 4: MLflow Experiment Tracking')
print('[INFO] =========================================')
subprocess.run([sys.executable,'src/train_models.py'],check=True)
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('Flight_Delay_Prediction')
with open('outputs/model_results.json') as f: results=json.load(f)
for r in results:
    with mlflow.start_run(run_name=r['Model']):
        mlflow.log_param('model',r['Model']); mlflow.log_param('dataset_rows',100000)
        for k in ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']: mlflow.log_metric(k.lower().replace(' ','_').replace('-','_'),r[k])
        mlflow.log_artifact('outputs/model_results.json')
print('[SUCCESS] MLflow tracking completed. Run `mlflow ui` to view experiments.')
