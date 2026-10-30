REQUIRED_FIELDS=['age','sex','chest_pain_type','resting_bp','cholesterol','fasting_bs','resting_ecg','max_hr','exercise_angina','oldpeak','st_slope']
VALID_RANGES={'age':(1,120),'resting_bp':(50,250),'cholesterol':(0,700),'max_hr':(50,250),'oldpeak':(-5,10)}
CATEGORIES={'sex':{'M','F'},'chest_pain_type':{'TA','ATA','NAP','ASY'},'resting_ecg':{'Normal','ST','LVH'},'exercise_angina':{'Y','N'},'st_slope':{'Up','Flat','Down'}}

def validate_extracted_data(data):
    missing=[f for f in REQUIRED_FIELDS if data.get(f) is None or data.get(f)=='']
    invalid=[]
    for f,(lo,hi) in VALID_RANGES.items():
        v=data.get(f)
        if v is None: continue
        try:
            if float(v)<lo or float(v)>hi: invalid.append({'field':f,'value':v,'expected':f'{lo} - {hi}'})
        except: invalid.append({'field':f,'value':v,'expected':'Numeric value'})
    for f,allowed in CATEGORIES.items():
        v=data.get(f)
        if v is not None and v not in allowed: invalid.append({'field':f,'value':v,'expected':', '.join(sorted(allowed))})
    fbs=data.get('fasting_bs')
    if fbs is not None and fbs not in (0,1): invalid.append({'field':'fasting_bs','value':fbs,'expected':'0 or 1'})
    return {'valid':not missing and not invalid,'missing_fields':missing,'invalid_fields':invalid}
