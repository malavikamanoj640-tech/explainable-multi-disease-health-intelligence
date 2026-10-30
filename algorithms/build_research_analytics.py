import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, os
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
from sklearn.metrics import roc_curve, brier_score_loss
from algorithms.adaptive_ensemble import AdaptiveEnsemble

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET=os.path.join(BASE_DIR,'dataset','heart.csv'); OUT=os.path.join(BASE_DIR,'research_analytics.json')
FEATURES=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']; TARGET='HeartDisease'
CAT=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']; NUM=[x for x in FEATURES if x not in CAT]

def prep():
    return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),CAT)])

def models():
    return [
      ('Logistic Regression', Pipeline([('prep',prep()),('model',LogisticRegression(max_iter=2000,random_state=42))])),
      ('Random Forest', Pipeline([('prep',prep()),('model',RandomForestClassifier(n_estimators=400,random_state=42,class_weight='balanced'))])),
      ('Gradient Boosting', Pipeline([('prep',prep()),('model',GradientBoostingClassifier(random_state=42))])),
      ('SVM', Pipeline([('prep',prep()),('model',SVC(probability=True,random_state=42,class_weight='balanced'))]))]

df=pd.read_csv(DATASET); X=df[FEATURES]; y=df[TARGET]
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
ms=models(); fitted=[]; curves={}; brier={}
for name,m in ms:
    m.fit(Xtr,ytr); fitted.append((name,m)); p=m.predict_proba(Xte)[:,1]
    fpr,tpr,_=roc_curve(yte,p)
    curves[name]={'fpr':[round(float(v),6) for v in fpr],'tpr':[round(float(v),6) for v in tpr]}
    brier[name]=round(float(brier_score_loss(yte,p)),6)
weights=np.array([0.900474,0.913462,0.901961,0.902913]); weights=np.maximum(weights,.01); weights/=weights.sum()
ad=AdaptiveEnsemble(fitted,weights,confidence_power=2.0)
ap=ad.predict_proba(Xte)[:,1]
fpr,tpr,_=roc_curve(yte,ap); curves['Adaptive Ensemble']={'fpr':[round(float(v),6) for v in fpr],'tpr':[round(float(v),6) for v in tpr]}; brier['Adaptive Ensemble']=round(float(brier_score_loss(yte,ap)),6)
probs=np.column_stack([m.predict_proba(Xte)[:,1] for _,m in fitted]); avg=probs.mean(axis=1)
fpr,tpr,_=roc_curve(yte,avg); curves['Static Probability Average']={'fpr':[round(float(v),6) for v in fpr],'tpr':[round(float(v),6) for v in tpr]}; brier['Static Probability Average']=round(float(brier_score_loss(yte,avg)),6)
report=json.load(open(os.path.join(BASE_DIR,'new_heart_model_report.json')))
validation_path=os.path.join(BASE_DIR,'research_validation_report.json')
if os.path.exists(validation_path):
    with open(validation_path,'r',encoding='utf-8') as vf:
        validation=json.load(vf)
    report['five_fold_cv_roc_auc']=validation.get('five_fold_cv_roc_auc',{})
report['roc_curves']=curves
report['brier_scores']=brier
report['evaluation_notes']=['ROC curves, confusion matrices and Brier scores use the same stratified 80/20 research split as the saved benchmark report.','Brier score measures probability prediction error; lower is better.','Predicted probabilities are model outputs and are not clinical diagnoses or guaranteed calibrated probabilities.']
with open(OUT,'w') as f: json.dump(report,f,indent=2)
print('Saved',OUT)
