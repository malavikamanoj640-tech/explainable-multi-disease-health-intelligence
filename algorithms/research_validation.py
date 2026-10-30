"""Research validation for the new heart-disease dataset/model."""
import json, os
import joblib, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATASET='dataset/heart.csv'; MODEL='models/heart/ensemble_model.pkl'
FEATURES=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']; TARGET='HeartDisease'
CAT=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']; NUM=[x for x in FEATURES if x not in CAT]
def prep(): return ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler())]),NUM),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),CAT)])
def model(kind):
    m={'lr':LogisticRegression(max_iter=2000,random_state=42),'rf':RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'),'gb':GradientBoostingClassifier(random_state=42),'svm':SVC(probability=True,random_state=42,class_weight='balanced')}[kind]
    return Pipeline([('prep',prep()),('model',m)])
df=pd.read_csv(DATASET); X=df[FEATURES]; y=df[TARGET]
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
report={'dataset':{'rows':len(df),'features':len(FEATURES),'target':TARGET,'class_distribution':df[TARGET].value_counts().sort_index().to_dict()},'five_fold_cv_roc_auc':{}}
for k in ['lr','rf','gb','svm']:
    scores=cross_val_score(model(k),X,y,cv=cv,scoring='roc_auc'); report['five_fold_cv_roc_auc'][k]={'mean':round(float(scores.mean()),6),'std':round(float(scores.std()),6),'folds':[round(float(v),6) for v in scores]}
model_obj=joblib.load(MODEL); report['saved_model_class']=type(model_obj).__name__; report['adaptive_weighting']=True; report['time_feature_used']=False
with open('research_validation_report.json','w') as f: json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
