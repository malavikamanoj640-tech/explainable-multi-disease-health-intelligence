import re

FIELDS = ["pregnancies","glucose","blood_pressure","skin_thickness","insulin","bmi","diabetes_pedigree_function","age"]

def _num(text, patterns):
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            try: return float(m.group(1))
            except: pass
    return None

def parse_diabetes_report(text):
    text=(text or "").replace("\xa0"," ").replace("\r","\n").replace("\t"," ")
    text=re.sub(r"[ ]{2,}"," ",text)
    d={f:None for f in FIELDS}
    d["pregnancies"]=_num(text,[r"\b(?:pregnancies|number\s+of\s+pregnancies)\s*[:=\-]\s*(\d+(?:\.\d+)?)"])
    d["glucose"]=_num(text,[r"\b(?:glucose|blood\s+glucose|fasting\s+glucose|plasma\s+glucose)\s*[:=\-]\s*(\d+(?:\.\d+)?)"])
    d["blood_pressure"]=_num(text,[r"\b(?:blood\s+pressure|diastolic\s+blood\s+pressure|bp)\s*[:=\-]\s*(\d+(?:\.\d+)?)"])
    d["skin_thickness"]=_num(text,[r"\b(?:skin\s+thickness|triceps\s+skin\s+thickness)\s*[:=\-]\s*(\d+(?:\.\d+)?)"])
    d["insulin"]=_num(text,[r"\b(?:insulin|serum\s+insulin)\s*[:=\-]\s*(\d+(?:\.\d+)?)"])
    d["bmi"]=_num(text,[r"\b(?:bmi|body\s+mass\s+index)\s*[:=\-]\s*(\d+(?:\.\d+)?)"])
    d["diabetes_pedigree_function"]=_num(text,[r"\b(?:diabetes\s+pedigree\s+function|pedigree\s+function|dpf)\s*[:=\-]\s*(\d+(?:\.\d+)?)"])
    d["age"]=_num(text,[r"\bage\s*[:=\-]?\s*(\d{1,3})\b"])
    ranges={"pregnancies":(0,30),"glucose":(20,500),"blood_pressure":(20,250),"skin_thickness":(0,150),"insulin":(0,1000),"bmi":(5,100),"diabetes_pedigree_function":(0,3),"age":(1,120)}
    for k,(lo,hi) in ranges.items():
        if d[k] is not None and not lo<=d[k]<=hi: d[k]=None
    return d
