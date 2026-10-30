"""Personalized, rule-based risk-reduction guidance for the heart module.

This module deliberately provides educational risk-reduction guidance rather than
medical diagnosis, prescriptions, or treatment plans. Rules are transparent and
based only on the values supplied to the prediction model and prediction history.
"""


def _add_unique(items, text):
    if text not in items:
        items.append(text)


def generate_recommendations(data, prediction):
    """Backward-compatible recommendation groups used by the existing result page."""
    r = {"Medical Advice": [], "Lifestyle": [], "Diet": [], "Follow-up": []}

    if data["resting_bp"] >= 140:
        r["Medical Advice"].append(
            "The recorded resting blood pressure is elevated; discuss it with a qualified healthcare professional."
        )
    else:
        r["Medical Advice"].append(
            "Continue monitoring blood pressure as part of routine preventive care."
        )

    if data["cholesterol"] >= 240:
        r["Diet"].append(
            "The recorded cholesterol is high; discuss diet, repeat testing, and overall cardiovascular risk with a healthcare professional."
        )
    else:
        r["Diet"].append(
            "Maintain a balanced diet rich in vegetables, whole grains and minimally processed foods."
        )

    if data["fasting_bs"] == 1:
        r["Diet"].append(
            "Fasting blood sugar is at or above the dataset threshold; discuss glucose monitoring with a healthcare professional."
        )

    if data["exercise_angina"] == "Y":
        r["Medical Advice"].append(
            "Exercise-related chest discomfort is recorded; discuss this finding with a healthcare professional, especially if symptoms recur."
        )

    if data["max_hr"] < 100:
        r["Follow-up"].append(
            "Discuss the recorded maximum heart-rate value with a healthcare professional in the context of the exercise test."
        )

    if data["oldpeak"] > 2:
        r["Medical Advice"].append(
            "The recorded Oldpeak value is elevated; it should be interpreted together with the ECG/exercise findings by a qualified professional."
        )

    if data["chest_pain_type"] in ("TA", "ASY"):
        r["Medical Advice"].append(
            "The recorded chest-pain category should be interpreted with the complete clinical history rather than in isolation."
        )

    r["Lifestyle"].append(
        "Avoid smoking and maintain regular physical activity appropriate for your health status."
    )

    if prediction == 1:
        r["Medical Advice"].append(
            "The model predicts a higher likelihood of heart disease in this dataset. Discuss the result and your measurements with a qualified healthcare professional; this is not a diagnosis."
        )
        r["Follow-up"].append(
            "Ask a qualified healthcare professional whether further cardiovascular assessment is appropriate."
        )
    else:
        r["Follow-up"].append(
            "Continue routine preventive care and discuss cardiovascular risk factors during regular health check-ups."
        )
    return r


def generate_personalized_plan(data, prediction, history=None, xai_explanation=None):
    """Create transparent personalized priorities and monitoring insights.

    `history` is expected to be a list of dictionaries containing at least
    prediction, confidence and prediction_date. XAI is used only as a supporting
    explanation signal; recommendations never claim that SHAP proves causation.
    """
    history = history or []
    priorities = []
    actions = []
    monitoring = []

    # Priority rules are intentionally transparent and tied to supplied inputs.
    if data["resting_bp"] >= 140:
        priorities.append({
            "level": "High attention",
            "factor": "Resting blood pressure",
            "value": f"{data['resting_bp']:.0f} mm Hg",
            "reason": "The submitted resting BP is at or above 140 mm Hg.",
            "action": "Track blood pressure readings consistently and discuss the pattern with a qualified healthcare professional."
        })
        actions.append("Prioritize blood-pressure monitoring and professional review of repeated elevated readings.")

    if data["cholesterol"] >= 240:
        priorities.append({
            "level": "High attention",
            "factor": "Cholesterol",
            "value": f"{data['cholesterol']:.0f} mg/dL",
            "reason": "The submitted cholesterol value is at or above 240 mg/dL.",
            "action": "Discuss the result and overall cardiovascular risk with a qualified healthcare professional."
        })
        actions.append("Favor a heart-healthy eating pattern and discuss whether repeat lipid testing is appropriate.")

    if data["fasting_bs"] == 1:
        priorities.append({
            "level": "Attention",
            "factor": "Fasting blood sugar",
            "value": "Dataset threshold reached",
            "reason": "The submitted fasting blood sugar indicator is 1.",
            "action": "Discuss glucose monitoring and interpretation with a qualified healthcare professional."
        })
        actions.append("Keep meals balanced and discuss glucose follow-up when appropriate.")

    if data["exercise_angina"] == "Y":
        priorities.append({
            "level": "High attention",
            "factor": "Exercise angina",
            "value": "Yes",
            "reason": "Exercise-related angina is recorded in the submitted assessment.",
            "action": "Discuss this finding with a qualified healthcare professional, particularly if symptoms recur or worsen."
        })
        actions.append("Do not use the app's risk score as permission to ignore exercise-related chest symptoms.")

    if data["oldpeak"] > 2:
        priorities.append({
            "level": "Attention",
            "factor": "Oldpeak",
            "value": f"{data['oldpeak']:.1f}",
            "reason": "The submitted Oldpeak value is above 2.",
            "action": "Have the value interpreted together with the exercise and ECG findings by a qualified professional."
        })

    if data["chest_pain_type"] == "ASY":
        priorities.append({
            "level": "Attention",
            "factor": "Chest pain category",
            "value": "ASY",
            "reason": "The submitted chest-pain category is ASY (asymptomatic in the dataset).",
            "action": "Interpret this coded field with the complete clinical history rather than in isolation."
        })

    if data["st_slope"] in ("Flat", "Down"):
        priorities.append({
            "level": "Attention",
            "factor": "ST slope",
            "value": data["st_slope"],
            "reason": "The submitted ST-slope category is Flat or Down.",
            "action": "Use the ECG-related information as part of professional interpretation rather than as a standalone diagnosis."
        })

    # General actions always remain available.
    _add_unique(actions, "Avoid smoking and second-hand smoke exposure where possible.")
    _add_unique(actions, "Choose regular, appropriate physical activity based on your health status and professional advice.")
    _add_unique(actions, "Maintain a balanced eating pattern with vegetables, whole grains, legumes and minimally processed foods.")

    # Prediction + history monitoring.
    high_count = sum(1 for h in history if str(h.get("prediction", "")).lower().startswith("high"))
    prior_count = len(history)
    if prediction == 1:
        monitoring.append("Current assessment is classified by the model as higher risk; review the model result together with your measurements rather than treating it as a diagnosis.")
    else:
        monitoring.append("Current assessment is classified by the model as lower risk; continue preventive monitoring because a lower model estimate does not guarantee absence of disease.")

    if prior_count:
        monitoring.append(f"This is assessment #{prior_count + 1} available in your saved history.")
        if high_count:
            monitoring.append(f"Your saved history contains {high_count} earlier higher-risk model result(s); look for persistent patterns with a qualified professional.")
        else:
            monitoring.append("No earlier higher-risk model result is recorded in the retrieved history.")
    else:
        monitoring.append("No earlier prediction history was available for comparison, so this assessment has no historical trend context yet.")

    if len(history) >= 2:
        latest_prob = float(history[0].get("high_risk_probability", history[0].get("confidence", 0)) or 0)
        previous_prob = float(history[1].get("high_risk_probability", history[1].get("confidence", 0)) or 0)
        delta = round(latest_prob - previous_prob, 2)
        monitoring.append(f"The previous two stored high-risk class probabilities differ by {abs(delta):.2f} percentage points; changes should be interpreted as model output changes, not as proof of a health change.")

    # SHAP is deliberately described as model behavior, never causation.
    xai_count = 0
    if xai_explanation and xai_explanation.get("available"):
        xai_count = len(xai_explanation.get("features", []))
        if xai_count:
            monitoring.append(f"The current report also contains {xai_count} local XAI feature attribution(s) that explain model behavior for this assessment.")

    if not priorities:
        priorities.append({
            "level": "Routine",
            "factor": "No rule-based priority trigger",
            "value": "Within configured thresholds",
            "reason": "None of the configured high-attention input thresholds was triggered.",
            "action": "Continue preventive care and review cardiovascular risk factors during routine health check-ups."
        })

    return {
        "priorities": priorities,
        "actions": actions,
        "monitoring": monitoring,
        "history_count": prior_count,
        "previous_high_count": high_count,
        "xai_supported": bool(xai_count),
        "disclaimer": "These recommendations are transparent, rule-based educational guidance derived from the submitted assessment and saved model history. They are not a diagnosis, prescription, or individualized treatment plan."
    }
