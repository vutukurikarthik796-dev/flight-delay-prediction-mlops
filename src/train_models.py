import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import json, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from common import load_clean, FEATURES, CAT, NUM, ARTIFACTS, MODELS, OUTPUTS

NROWS = 100000
print(f'[INFO] Loading up to {NROWS:,} rows for reproducible training...')
df=load_clean(NROWS)
X,y=df[FEATURES],df['DELAYED']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
pre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),NUM),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),CAT)])
models={
'Logistic Regression':LogisticRegression(max_iter=1000,class_weight='balanced'),
'Decision Tree':DecisionTreeClassifier(max_depth=15,min_samples_split=20,random_state=42,class_weight='balanced'),
'Random Forest':RandomForestClassifier(n_estimators=100,max_depth=15,min_samples_split=20,random_state=42,n_jobs=-1,class_weight='balanced'),
'Gradient Boosting':GradientBoostingClassifier(n_estimators=100,learning_rate=.1,max_depth=5,random_state=42),
'XGBoost':XGBClassifier(n_estimators=100,max_depth=6,learning_rate=.1,subsample=.8,colsample_bytree=.8,random_state=42,eval_metric='logloss',n_jobs=-1)
}
results=[]
for name,clf in models.items():
    print(f'[INFO] Training {name}...')
    pipe=Pipeline([('preprocessor',pre),('classifier',clf)])
    pipe.fit(X_train,y_train); pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
    m={'Model':name,'Accuracy':accuracy_score(y_test,pred),'Precision':precision_score(y_test,pred,zero_division=0),'Recall':recall_score(y_test,pred,zero_division=0),'F1 Score':f1_score(y_test,pred,zero_division=0),'ROC-AUC':roc_auc_score(y_test,prob)}
    results.append(m)
    safe=name.lower().replace(' ','_')+'.pkl'; joblib.dump(pipe,MODELS/safe)
joblib.dump((X_train,X_test,y_train,y_test),ARTIFACTS/'training_split.pkl')
with open(OUTPUTS/'model_results.json','w') as f: json.dump(results,f,indent=2)
best=max(results,key=lambda x:x['F1 Score'])
joblib.dump(models and joblib.load(MODELS/(best['Model'].lower().replace(' ','_')+'.pkl')), MODELS/'best_model.pkl')
with open(ARTIFACTS/'best_model_info.json','w') as f: json.dump(best,f,indent=2)
print('\n[INFO] Model comparison:')
for r in results: print(r['Model'], {k:round(r[k],4) for k in ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']})
print(f"[SUCCESS] Best model by F1: {best['Model']}")
