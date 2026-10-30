HEART_REQUIRED_FIELDS=['age','sex','chest_pain_type','resting_bp','cholesterol','fasting_bs','resting_ecg','max_hr','exercise_angina','oldpeak','st_slope']
def check_heart_model_compatibility(data):
    missing=[f for f in HEART_REQUIRED_FIELDS if data.get(f) is None or data.get(f)=='']
    total=len(HEART_REQUIRED_FIELDS); available=total-len(missing)
    return {'compatible':not missing,'missing_fields':missing,'available_fields':available,'required_fields':total,'completion_percentage':round(available/total*100,2)}
