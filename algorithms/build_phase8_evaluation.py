"""Phase 8: research-grade evaluation for the heart disease models.

Uses repeated stratified 5-fold outer evaluation. Adaptive ensemble weights are
computed only from an inner 3-fold validation on each outer training split,
which avoids using the outer test fold to choose weights.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from algorithms.adaptive_ensemble import AdaptiveEnsemble

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET=os.path.join(BASE_DIR,'dataset','heart.csv')
OUT=os.path.join(BASE_DIR,'phase8_research_evaluation.json')
FEATURES=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
TARGET='HeartDisease'
CAT=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
NUM=[x for x in FEATURES if x not in CAT]
NAMES=['Logistic Regression','Random Forest','Gradient Boosting','SVM']
KEYS=['lr','rf','gb','svm']

def prep():
    return ColumnTransformer([
        ('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),
        ('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),CAT)
    ])

def make_models():
    return [
      ('Logistic Regression',Pipeline([('prep',prep()),('model',LogisticRegression(max_iter=2000,random_state=42))])),
      ('Random Forest',Pipeline([('prep',prep()),('model',RandomForestClassifier(n_estimators=250,random_state=42,class_weight='balanced'))])),
      ('Gradient Boosting',Pipeline([('prep',prep()),('model',GradientBoostingClassifier(random_state=42))])),
      ('SVM',Pipeline([('prep',prep()),('model',SVC(probability=True,random_state=42,class_weight='balanced'))]))
    ]

def metric_dict(y,p,prob):
    return {
      'accuracy':float(accuracy_score(y,p)),
      'precision':float(precision_score(y,p,zero_division=0)),
      'recall':float(recall_score(y,p,zero_division=0)),
      'f1':float(f1_score(y,p,zero_division=0)),
      'roc_auc':float(roc_auc_score(y,prob)),
      'brier':float(brier_score_loss(y,prob))
    }

def inner_weights(X,y,seed):
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
    scores=[]
    for name,model in make_models():
        fold=[]
        for tr,va in cv.split(X,y):
            model.fit(X.iloc[tr],y.iloc[tr])
            pred=model.predict(X.iloc[va])
            fold.append(f1_score(y.iloc[va],pred,zero_division=0))
        scores.append(np.mean(fold))
    w=np.maximum(np.asarray(scores,float),0.01)
    w=w/w.sum()
    return w

def mean_std(vals):
    return {'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=0))}

df=pd.read_csv(DATASET); X=df[FEATURES]; y=df[TARGET]
outer_results={k:[] for k in KEYS+['static','adaptive']}
pooled={k:[] for k in KEYS+['static','adaptive']}
weight_rows=[]
outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
# Three deterministic repeats provide a more stable estimate than one split.
for repeat in range(3):
    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=42+repeat)
    for fold,(tr,te) in enumerate(outer.split(X,y),1):
        Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y.iloc[tr],y.iloc[te]
        fitted=[]
        for name,model in make_models():
            model.fit(Xtr,ytr); fitted.append((name,model))
            prob=model.predict_proba(Xte)[:,1]; pred=(prob>=0.5).astype(int)
            key=KEYS[NAMES.index(name)]
            outer_results[key].append(metric_dict(yte,pred,prob))
            pooled[key].extend(zip(yte.tolist(),prob.tolist()))
        probs=np.column_stack([m.predict_proba(Xte)[:,1] for _,m in fitted])
        avg=probs.mean(axis=1); pred=(avg>=.5).astype(int)
        outer_results['static'].append(metric_dict(yte,pred,avg)); pooled['static'].extend(zip(yte.tolist(),avg.tolist()))
        w=inner_weights(Xtr.reset_index(drop=True),ytr.reset_index(drop=True),seed=1000+repeat*10+fold)
        adaptive=AdaptiveEnsemble(fitted,w,confidence_power=2.0)
        ap=adaptive.predict_proba(Xte)[:,1]; pred=(ap>=.5).astype(int)
        outer_results['adaptive'].append(metric_dict(yte,pred,ap)); pooled['adaptive'].extend(zip(yte.tolist(),ap.tolist()))
        weight_rows.append({'repeat':repeat+1,'fold':fold,'weights':{n:float(v) for n,v in zip(NAMES,w)}})

summary={}
for key in KEYS+['static','adaptive']:
    summary[key]={m:mean_std([r[m] for r in outer_results[key]]) for m in ['accuracy','precision','recall','f1','roc_auc','brier']}
    ypool=np.array([x[0] for x in pooled[key]]); ppool=np.array([x[1] for x in pooled[key]])
    frac,mean_pred=calibration_curve(ypool,ppool,n_bins=10,strategy='uniform')
    summary[key]['pooled_brier']=float(brier_score_loss(ypool,ppool))
    summary[key]['calibration']={'mean_predicted':[float(v) for v in mean_pred],'observed_fraction':[float(v) for v in frac]}

# Explicit research comparisons, without claiming adaptive superiority.
summary['comparisons']={
 'adaptive_vs_static':{m:float(summary['adaptive'][m]['mean']-summary['static'][m]['mean']) for m in ['accuracy','precision','recall','f1','roc_auc','brier']},
 'interpretation':'Positive differences indicate a higher adaptive mean for that metric; for Brier score, negative is preferable.'
}
report={
 'dataset':{'rows':int(len(df)),'features':len(FEATURES),'target':TARGET,'features_list':FEATURES,'evaluation_protocol':'Repeated stratified 5-fold outer evaluation with 3 repeats; adaptive weights selected using an inner 3-fold validation on each outer training split.'},
 'repeats':3,'outer_folds':5,'inner_folds_for_adaptive_weights':3,
 'models':summary,
 'adaptive_weight_samples':weight_rows,
 'notes':[
  'Outer test folds are not used to select adaptive ensemble weights.',
  'Metrics summarize repeated cross-validation and should not be interpreted as clinical performance.',
  'Brier score evaluates probability error; lower values are better.',
  'Calibration curves describe model probability behavior on this research dataset, not clinical risk calibration.',
  'The evaluation compares static probability averaging with sample-adaptive weighting without assuming adaptive weighting is superior.'
 ]
}
with open(OUT,'w',encoding='utf-8') as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
print('Saved',OUT)
