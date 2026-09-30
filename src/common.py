from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ARTIFACTS = ROOT / 'artifacts'
MODELS = ROOT / 'models'
OUTPUTS = ROOT / 'outputs'
for p in (ARTIFACTS, MODELS, OUTPUTS): p.mkdir(exist_ok=True)

FEATURES = ['MONTH','DAY','DAY_OF_WEEK','AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT','SCHEDULED_TIME','DISTANCE','DEP_HOUR','DEP_MINUTE','DEP_PERIOD']
CAT = ['AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT','DEP_PERIOD']
NUM = ['MONTH','DAY','DAY_OF_WEEK','SCHEDULED_TIME','DISTANCE','DEP_HOUR','DEP_MINUTE']

def load_clean(nrows=None):
    df = pd.read_csv(
    DATA / 'flights.csv',
    nrows=nrows,
    low_memory=False,
    dtype={
        'ORIGIN_AIRPORT': str,
        'DESTINATION_AIRPORT': str
    }
    )
    df = df.drop_duplicates()
    df = df[df['CANCELLED'] == 0].copy()
    df = df.dropna(subset=['ARRIVAL_DELAY'])
    df['DELAYED'] = (df['ARRIVAL_DELAY'] > 15).astype(int)
    df['DEP_HOUR'] = df['SCHEDULED_DEPARTURE'] // 100
    df['DEP_MINUTE'] = df['SCHEDULED_DEPARTURE'] % 100
    df['DEP_PERIOD'] = pd.cut(df['DEP_HOUR'], [-1,5,11,17,23], labels=['Night','Morning','Afternoon','Evening']).astype(str)
    return df
