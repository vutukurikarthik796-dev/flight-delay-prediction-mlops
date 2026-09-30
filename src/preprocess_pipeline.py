import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from common import load_clean, FEATURES, CAT, NUM, ARTIFACTS

print('[INFO] Starting Sklearn Pipeline Preprocessing...')
df = load_clean()
X, y = df[FEATURES], df['DELAYED']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
num_pipe = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
cat_pipe = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))])
pre = ColumnTransformer([('num',num_pipe,NUM),('cat',cat_pipe,CAT)])
print('[INFO] Fitting and transforming training data...')
Xtr = pre.fit_transform(X_train)
print('[INFO] Transforming test data...')
Xte = pre.transform(X_test)
joblib.dump(pre, ARTIFACTS/'preprocessor.pkl')
joblib.dump((X_train,X_test,y_train,y_test), ARTIFACTS/'splits.pkl')
print('[SUCCESS] Preprocessing completed successfully! Pipeline saved as artifacts/preprocessor.pkl.')
print(f'[INFO] Train matrix: {Xtr.shape}; Test matrix: {Xte.shape}')
