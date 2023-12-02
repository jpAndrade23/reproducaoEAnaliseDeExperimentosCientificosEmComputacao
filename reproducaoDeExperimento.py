import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import  auc, roc_curve
from sklearn.metrics import auc 
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import time

data = pd.read_csv("data.csv")

#data['income'] = data['income'].apply(lambda x: 0 if x=='<=50k' else 1)

#parâmetros usados no artigo
value_depth = 6
value_learn_rate = 0.1
SEED = 42
value_estimators = [100, 250, 500]

le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = le.fit_transform(data[col])


X = data.drop('income', axis=1)
y = data['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=SEED, stratify=y)

aucs_xgb = []
aucs_ctb = []
time_trainCtb = []
time_trainXG = []
time_testCtb = []
time_testXG = []

for n_estimators in value_estimators:
    
    cls_xgb = xgb.XGBClassifier(random_state=SEED, learning_rate=value_learn_rate, objective='binary:logistic', max_depth=value_depth, n_estimators=n_estimators)
    cls_ctb = CatBoostClassifier(iterations=n_estimators, random_seed=SEED, depth=value_depth, learning_rate=value_learn_rate)
    
    timeTrain = time.time()
    ctb_classification = cls_ctb.fit(X_train, y_train)
    time_trainCtb.append(time.time() - timeTrain)
    
    timeTrainXG = time.time()
    xgb_classification = cls_xgb.fit(X_train, y_train)
    time_trainXG.append(time.time() - timeTrainXG )

    
    timeXG = time.time()
    probs = xgb_classification.predict_proba(X_test)[:, 1]
    time_testXG.append(time.time() - timeXG)

    timeCtb = time.time()
    probs1 = ctb_classification.predict_proba(X_test)[:, 1]
    time_testCtb.append(time.time() - timeCtb)
    
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    fpr1, tpr1, thresholds1 = roc_curve(y_test, probs1)

   
    a = auc(fpr, tpr)
    b = auc(fpr1, tpr1)

    aucs_xgb.append(a)
    aucs_ctb.append(b)


results = pd.DataFrame({
    'n_estimators': value_estimators,
    'auc_xgb': aucs_xgb,
    'auc_ctb': aucs_ctb,
    'train_time_ctb': time_trainCtb,
    'train_time_xgb': time_trainXG,
    'test_time_ctb': time_testCtb,
    'test_time_xgb': time_testXG
})

print(results)