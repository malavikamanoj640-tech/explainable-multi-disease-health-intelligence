import json, os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from algorithms.adaptive_ensemble import AdaptiveEnsemble

DATASET='dataset/heart.csv'
MODEL_PATH='models/heart/ensemble_model.pkl'
REPORT_PATH='new_heart_model_report.json'
TARGET='HeartDisease'
FEATURES=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
CAT=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
NUM=[x for x in FEATURES if x not in CAT]

def prep():
    return ColumnTransformer([
        ('num', Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]), NUM),
        ('cat', Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]), CAT)
    ])

def make_models():
    return [
      ('Logistic Regression', Pipeline([('prep',prep()),('model',LogisticRegression(max_iter=2000, random_state=42))])),
      ('Random Forest', Pipeline([('prep',prep()),('model',RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced'))])),
      ('Gradient Boosting', Pipeline([('prep',prep()),('model',GradientBoostingClassifier(random_state=42))])),
      ('SVM', Pipeline([('prep',prep()),('model',SVC(probability=True, random_state=42, class_weight='balanced'))]))
    ]

def metrics(y,p,proba):
    return {'accuracy':round(accuracy_score(y,p),6),'precision':round(precision_score(y,p,zero_division=0),6),'recall':round(recall_score(y,p,zero_division=0),6),'f1':round(f1_score(y,p,zero_division=0),6),'roc_auc':round(roc_auc_score(y,proba),6),'confusion_matrix':confusion_matrix(y,p).tolist()}

df=pd.read_csv(DATASET)
assert list(df.columns)==FEATURES+[TARGET], df.columns.tolist()
X=df[FEATURES]; y=df[TARGET]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
models=make_models(); fitted=[]; individual={}; f1s=[]
for name,m in models:
    m.fit(X_train,y_train); fitted.append((name,m))
    pr=m.predict(X_test); pp=m.predict_proba(X_test)[:,1]
    individual[name]=metrics(y_test,pr,pp); f1s.append(individual[name]['f1'])
# Normalize validation F1 into global reliability weights.
weights=np.array(f1s,float); weights=np.maximum(weights,0.01); weights=weights/weights.sum()
adaptive=AdaptiveEnsemble(fitted,weights,confidence_power=2.0)
pr=adaptive.predict(X_test); pp=adaptive.predict_proba(X_test)[:,1]
adaptive_metrics=metrics(y_test,pr,pp)
# Static average for direct research comparison.
probs=np.column_stack([m.predict_proba(X_test)[:,1] for _,m in fitted]); avg=probs.mean(axis=1); sp=(avg>=.5).astype(int)
static_metrics=metrics(y_test,sp,avg)
# Fit final base estimators on all available data and build final ensemble.
final_models=make_models()
for _,m in final_models: m.fit(X,y)
final_adaptive=AdaptiveEnsemble(final_models,weights,confidence_power=2.0)
final_adaptive.fit(X,y)
os.makedirs(os.path.dirname(MODEL_PATH),exist_ok=True)
joblib.dump(final_adaptive,MODEL_PATH)
report={'dataset':{'rows':len(df),'features':len(FEATURES),'target':TARGET,'class_distribution':df[TARGET].value_counts().sort_index().to_dict(),'features':FEATURES},'individual_models':individual,'static_probability_average':static_metrics,'adaptive_ensemble':adaptive_metrics,'validation_weights':{name:round(float(w),6) for (name,_),w in zip(fitted,weights)},'notes':['No follow-up-time feature is used.','Adaptive weights combine validation F1 reliability with per-sample model confidence.','Metrics are from one stratified 80/20 split and should not be presented as clinical performance.']}
with open(REPORT_PATH,'w') as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
print('Saved',MODEL_PATH)
