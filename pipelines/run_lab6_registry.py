import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, joblib, mlflow
from mlflow.models import infer_signature
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import MODELS, OUTPUTS, ARTIFACTS
print('[INFO] =========================================')
print('[INFO] Starting Lab 6: Model Registry')
print('[INFO] =========================================')
model=joblib.load(MODELS/'best_model.pkl')
with open(ARTIFACTS/'training_split.pkl','rb') as f: X_train,X_test,y_train,y_test=joblib.load(f)
with open(ARTIFACTS/'best_model_info.json') as f: info=json.load(f)
mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('Flight_Delay_Registry')
with mlflow.start_run(run_name='Best_Flight_Delay_Model') as run:
 mlflow.log_params({'model':info['Model'],'source':'mlops_158.ipynb'})
 for k in ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']: mlflow.log_metric(k.lower().replace(' ','_').replace('-','_'),info[k])
 sig=infer_signature(X_train,model.predict(X_train.head(5)))
 mlflow.sklearn.log_model(
    model,
    name='model',
    signature=sig,
    registered_model_name='FlightDelayModel',
    skops_trusted_types=[
        'numpy.dtype',
        'sklearn.tree._tree.Tree'
    ]
)
print('[SUCCESS] Model logged and registered as FlightDelayModel.')
print('[INFO] Start MLflow UI with: mlflow ui')
