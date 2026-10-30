"""SHAP KernelExplainer for the new 918-row heart disease classification model."""
import os,warnings
import numpy as np,pandas as pd
try: import shap
except ImportError: shap=None
FEATURE_LABELS={'Age':'Age','Sex':'Sex','ChestPainType':'Chest Pain Type','RestingBP':'Resting Blood Pressure','Cholesterol':'Cholesterol','FastingBS':'Fasting Blood Sugar','RestingECG':'Resting ECG','MaxHR':'Maximum Heart Rate','ExerciseAngina':'Exercise Angina','Oldpeak':'Oldpeak','ST_Slope':'ST Slope'}
FEATURE_EXPLANATIONS={k:f'{v} is supplied to the machine-learning model.' for k,v in FEATURE_LABELS.items()}
_BACKGROUND_CACHE=None; _EXPLAINER_CACHE=None
_MODEL_COLUMNS=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']

def _load_background(model_path):
    global _BACKGROUND_CACHE
    if _BACKGROUND_CACHE is not None:return _BACKGROUND_CACHE
    path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'dataset','heart.csv')
    df=pd.read_csv(path)
    X=df.drop('HeartDisease',axis=1)
    _BACKGROUND_CACHE=shap.sample(X,min(30,len(X)),random_state=42)
    return _BACKGROUND_CACHE

def _get_explainer(model,model_path):
    global _EXPLAINER_CACHE
    if _EXPLAINER_CACHE is None:
        _EXPLAINER_CACHE=shap.KernelExplainer(lambda x: model.predict_proba(pd.DataFrame(x, columns=_MODEL_COLUMNS)),_load_background(model_path),link='identity')
    return _EXPLAINER_CACHE

def explain_prediction(model,input_df,prediction_class,model_path='models/heart/ensemble_model.pkl'):
    if shap is None:return {'available':False,'error':'SHAP is not installed.','features':[]}
    try:
        ex=_get_explainer(model,model_path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore'); sv=ex.shap_values(input_df,nsamples=100)
        if isinstance(sv,list): vals=np.asarray(sv[1])[0]
        else:
            arr=np.asarray(sv); vals=arr[0,:,1] if arr.ndim==3 else arr[0]
        base=np.asarray(ex.expected_value); baseline=float(base[1] if base.ndim else base)
        rows=[]
        for feature,contribution in zip(input_df.columns,vals):
            value=input_df.iloc[0][feature]
            rows.append({'feature':feature,'label':FEATURE_LABELS.get(feature,feature),'value':str(value),'shap_value':round(float(contribution),5),'direction':'increases' if contribution>0 else 'decreases' if contribution<0 else 'neutral','magnitude':round(abs(float(contribution)),5),'description':FEATURE_EXPLANATIONS.get(feature,'Feature supplied to the model.')})
        rows.sort(key=lambda x:x['magnitude'],reverse=True)
        return {'available':True,'method':'SHAP KernelExplainer','target':'Class 1 (HeartDisease = 1)','baseline':round(baseline,5),'summary':'Positive SHAP contributions move the model output toward the HeartDisease=1 class; negative contributions move it away.','features':rows[:8],'all_features':rows,'positive_count':sum(x['shap_value']>0 for x in rows),'negative_count':sum(x['shap_value']<0 for x in rows)}
    except Exception as e:
        print('SHAP EXPLANATION ERROR:',e); return {'available':False,'error':str(e),'features':[]}
