import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import load_clean, DATA

df = load_clean()
required = ['ARRIVAL_DELAY','CANCELLED','SCHEDULED_DEPARTURE','AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT','SCHEDULED_TIME','DISTANCE']
missing = [c for c in required if c not in df.columns]
print(f'[INFO] Validating dataset (Records: {len(df):,})...')
if missing: raise ValueError(f'Missing required columns: {missing}')
if df.empty: raise ValueError('Dataset is empty after cleaning')
print('[SUCCESS] Schema Validation PASSED. Dataset is clean.')
