import re

FIELD_ALIASES = {
    'age':['age','patient age','age in years'],
    'sex':['sex','gender','patient sex','patient gender'],
    'chest_pain_type':['chest pain type','chest pain','chestpain type','cp type'],
    'resting_bp':['resting bp','resting blood pressure','restingbp','blood pressure','bp'],
    'cholesterol':['cholesterol','serum cholesterol','chol'],
    'fasting_bs':['fasting bs','fasting blood sugar','fasting glucose','fbs'],
    'resting_ecg':['resting ecg','resting electrocardiogram','ecg'],
    'max_hr':['max hr','maximum heart rate','max heart rate','maximum hr'],
    'exercise_angina':['exercise angina','exercise induced angina','exercise-induced angina'],
    'oldpeak':['oldpeak','old peak','st depression','st depression oldpeak'],
    'st_slope':['st slope','st segment slope','st-segment slope'],
}

def normalize_label(label):
    return re.sub(r'\s+',' ',str(label).lower().strip().replace('_',' ').replace('-',' '))

def get_standard_field(label):
    n=normalize_label(label)
    for field,aliases in FIELD_ALIASES.items():
        if n in [normalize_label(a) for a in aliases]: return field
    return None

def normalize_sex(v):
    s=str(v).strip().lower()
    if s in ('m','male','man','1'): return 'M'
    if s in ('f','female','woman','0'): return 'F'
    return None

def normalize_binary(v):
    s=str(v).strip().lower()
    if s in ('1','yes','y','true','positive','present'): return 1
    if s in ('0','no','n','false','negative','absent'): return 0
    return None

def normalize_category(v, field):
    s=str(v).strip().upper()
    maps={
      'chest_pain_type': {'TA':'TA','TYPICAL ANGINA':'TA','ATA':'ATA','ATYPICAL ANGINA':'ATA','NAP':'NAP','NON-ANGINAL PAIN':'NAP','ASY':'ASY','ASYMPTOMATIC':'ASY'},
      'resting_ecg': {'NORMAL':'Normal','ST':'ST','LVH':'LVH','LEFT VENTRICULAR HYPERTROPHY':'LVH'},
      'exercise_angina': {'Y':'Y','YES':'Y','N':'N','NO':'N'},
      'st_slope': {'UP':'Up','FLAT':'Flat','DOWN':'Down'},
    }
    return maps.get(field,{}).get(s)

def normalize_extracted_data(data):
    out={f:None for f in FIELD_ALIASES}
    for key,value in data.items():
        field=get_standard_field(key) or (key if key in out else None)
        if not field: continue
        if field=='sex': out[field]=normalize_sex(value)
        elif field in ('fasting_bs','exercise_angina'): out[field]=normalize_binary(value) if field=='fasting_bs' else normalize_category(value,field)
        elif field in ('chest_pain_type','resting_ecg','st_slope'): out[field]=normalize_category(value,field)
        else:
            try:
                n=float(value); out[field]=int(n) if n.is_integer() else n
            except (TypeError,ValueError): out[field]=None
    return out
