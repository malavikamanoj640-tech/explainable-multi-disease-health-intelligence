import re
from algorithms.report_normalizer import normalize_binary

EXPECTED_FIELDS={f:None for f in ['age','sex','chest_pain_type','resting_bp','cholesterol','fasting_bs','resting_ecg','max_hr','exercise_angina','oldpeak','st_slope']}

def clean_text(text):
    text=(text or '').replace('\xa0',' ').replace('\r','\n').replace('\t',' ')
    text=re.sub(r'\.{2,}',':',text); text=re.sub(r'[ ]{2,}',' ',text); return text.strip()

def num(text,patterns):
    for p in patterns:
        m=re.search(p,text,re.I)
        if m:
            try:return float(m.group(1))
            except:pass
    return None

def value(text,patterns):
    for p in patterns:
        m=re.search(p,text,re.I)
        if m:return m.group(1)
    return None

def parse_report(text):
    text=clean_text(text); d=EXPECTED_FIELDS.copy()
    d['age']=num(text,[r'\bage\s*[:=\-]?\s*(\d{1,3})\b'])
    sx=value(text,[r'\b(?:sex|gender)\s*[:=\-]\s*(male|female|m|f)\b',r'\b(?:sex|gender)\s+(male|female|m|f)\b'])
    d['sex']='M' if sx and sx.lower() in ('m','male') else 'F' if sx else None
    cp=value(text,[r'\bchest\s*pain\s*(?:type)?\s*[:=\-]\s*(TA|ATA|NAP|ASY)\b'])
    d['chest_pain_type']=cp.upper() if cp else None
    d['resting_bp']=num(text,[r'\b(?:resting\s*bp|resting\s*blood\s*pressure)\s*[:=\-]\s*(\d+(?:\.\d+)?)',r'\bblood\s*pressure\s*[:=\-]\s*(\d{2,3})'])
    d['cholesterol']=num(text,[r'\b(?:cholesterol|serum\s+cholesterol)\s*[:=\-]\s*(\d+(?:\.\d+)?)'])
    fbs=value(text,[r'\b(?:fasting\s*bs|fasting\s*blood\s*sugar|fbs)\s*[:=\-]\s*(yes|no|0|1|true|false)\b'])
    d['fasting_bs']=normalize_binary(fbs) if fbs else None
    ecg=value(text,[r'\b(?:resting\s*ecg|resting\s*electrocardiogram|ecg)\s*[:=\-]\s*(Normal|ST|LVH)\b'])
    d['resting_ecg']=ecg.title() if ecg and ecg.lower()=='normal' else ecg.upper() if ecg else None
    d['max_hr']=num(text,[r'\b(?:max\s*hr|maximum\s*heart\s*rate|maximum\s*hr)\s*[:=\-]\s*(\d+(?:\.\d+)?)'])
    ea=value(text,[r'\b(?:exercise\s*angina|exercise\s*induced\s*angina)\s*[:=\-]\s*(yes|no|y|n)\b'])
    d['exercise_angina']='Y' if ea and ea.lower() in ('yes','y') else 'N' if ea else None
    d['oldpeak']=num(text,[r'\bold\s*peak\s*[:=\-]\s*(-?\d+(?:\.\d+)?)',r'\boldpeak\s*[:=\-]\s*(-?\d+(?:\.\d+)?)'])
    slope=value(text,[r'\bst\s*slope\s*[:=\-]\s*(up|flat|down)\b'])
    d['st_slope']=slope.title() if slope else None
    # Clean numeric types and conservative ranges.
    if d['age'] is not None and not 1<=d['age']<=120:d['age']=None
    if d['resting_bp'] is not None and not 50<=d['resting_bp']<=250:d['resting_bp']=None
    if d['cholesterol'] is not None and not 0<=d['cholesterol']<=700:d['cholesterol']=None
    if d['max_hr'] is not None and not 50<=d['max_hr']<=250:d['max_hr']=None
    if d['oldpeak'] is not None and not -5<=d['oldpeak']<=10:d['oldpeak']=None
    return d
