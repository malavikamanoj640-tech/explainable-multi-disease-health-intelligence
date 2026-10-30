"""Ollama-powered AI Health Assistant.

Uses deterministic responses for application-specific prediction questions
and Ollama for general health education.

The assistant must not diagnose, prescribe medicines/doses, provide treatment
plans, or replace a qualified healthcare professional.
"""

import re
import requests


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:0.5b"


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the AI Health Assistant inside a multi-disease health prediction
application.

Give short, simple and educational health information.

Never diagnose a disease.
Never prescribe medicines or doses.
Never give personalized treatment plans.
Never claim certainty.
Never change or override the application's machine-learning prediction.
Do not pretend to be a doctor.

For personal medical decisions, recommend consulting a qualified healthcare
professional.

For possible emergencies such as severe chest pain, severe breathing
difficulty, fainting, sudden weakness, or sudden severe symptoms, advise
urgent medical care.

When discussing an application's prediction, clearly explain that it is a
machine-learning risk estimate and not a medical diagnosis.
""".strip()


# ============================================================
# FIELD LABELS
# ============================================================

FIELD_LABELS = {
    "age":"Age", "sex":"Sex (M/F)", "chest_pain_type":"Chest pain type",
    "resting_bp":"Resting blood pressure (mm Hg)", "cholesterol":"Cholesterol (mg/dL)",
    "fasting_bs":"Fasting blood sugar encoding (0/1)", "resting_ecg":"Resting ECG",
    "max_hr":"Maximum heart rate", "exercise_angina":"Exercise-induced angina (Y/N)",
    "oldpeak":"Oldpeak", "st_slope":"ST slope",
    "pregnancies":"Pregnancies", "glucose":"Glucose (mg/dL)",
    "blood_pressure":"Blood pressure (mmHg)", "skin_thickness":"Skin thickness (mm)",
    "insulin":"Insulin (µU/mL)", "bmi":"BMI (kg/m²)",
    "diabetes_pedigree_function":"Diabetes Pedigree Function",
    "age":"Age"
}


# ============================================================
# TEXT NORMALISATION
# ============================================================

def _normalise(text: str) -> str:
    """
    Convert text into a simple searchable format.
    """

    text = str(text or "").lower()

    text = re.sub(
        r"[^a-z0-9% ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DISEASE-SPECIFIC ASSISTANT HELPERS
# ============================================================

HEART_FIELDS = [
    ("age", "Age", "years"),
    ("sex", "Sex", ""),
    ("chest_pain_type", "Chest pain type", ""),
    ("resting_bp", "Resting blood pressure", "mmHg"),
    ("cholesterol", "Cholesterol", "mg/dL"),
    ("fasting_bs", "Fasting blood sugar", ""),
    ("resting_ecg", "Resting ECG", ""),
    ("max_hr", "Maximum heart rate", "bpm"),
    ("exercise_angina", "Exercise-induced angina", ""),
    ("oldpeak", "Oldpeak", ""),
    ("st_slope", "ST slope", ""),
]

DIABETES_FIELDS_DISPLAY = [
    ("pregnancies", "Pregnancies", ""),
    ("glucose", "Glucose", "mg/dL"),
    ("blood_pressure", "Blood pressure", "mmHg"),
    ("skin_thickness", "Skin thickness", "mm"),
    ("insulin", "Insulin", "µU/mL"),
    ("bmi", "BMI", "kg/m²"),
    ("diabetes_pedigree_function", "Diabetes Pedigree Function", ""),
    ("age", "Age", "years"),
]

EARLY_DIABETES_FIELDS = [
    ("age", "Age", "years"),
    ("gender", "Gender", ""),
    ("polyuria", "Polyuria", ""),
    ("polydipsia", "Polydipsia", ""),
    ("sudden_weight_loss", "Sudden weight loss", ""),
    ("weakness", "Weakness", ""),
    ("polyphagia", "Polyphagia", ""),
    ("genital_thrush", "Genital thrush", ""),
    ("visual_blurring", "Visual blurring", ""),
    ("itching", "Itching", ""),
    ("irritability", "Irritability", ""),
    ("delayed_healing", "Delayed healing", ""),
    ("partial_paresis", "Partial paresis", ""),
    ("muscle_stiffness", "Muscle stiffness", ""),
    ("alopecia", "Alopecia", ""),
    ("obesity", "Obesity", ""),
]


def _disease_request(message: str) -> tuple[str | None, str | None]:
    """Return (disease, intent) for explicit disease-specific requests."""
    text = _normalise(message)

    # Early Diabetes MUST be checked before ordinary Diabetes.
    early = any(p in text for p in [
        "early diabetes", "early diabetic", "early diabetes analysis",
        "early diabetes assessment", "early diabetes prediction",
    ])
    if early:
        if any(p in text for p in [
            "what values", "what inputs", "what data", "values used",
            "inputs used", "what did i enter", "what information was used",
            "show my inputs", "show input values", "show values used",
        ]):
            return "early_diabetes", "inputs"
        if any(p in text for p in ["why", "explain my", "explain this", "what caused"]):
            return "early_diabetes", "why"
        if any(p in text for p in ["factors", "what affects", "risk factors"]):
            return "early_diabetes", "factors"
        return "early_diabetes", "latest"

    heart = any(p in text for p in [
        "heart disease", "heart prediction", "heart assessment", "heart result",
        "heart risk", "cardiac prediction", "cardiac assessment",
    ])
    if heart:
        if any(p in text for p in [
            "what values", "what inputs", "what data", "values used",
            "inputs used", "what did i enter", "what information was used",
            "show my inputs", "show input values", "show values used",
        ]):
            return "heart", "inputs"
        if any(p in text for p in ["why", "explain my", "explain this", "what caused"]):
            return "heart", "why"
        if any(p in text for p in ["factors", "what affects", "risk factors"]):
            return "heart", "factors"
        return "heart", "latest"

    diabetes = any(p in text for p in [
        "diabetes", "diabetic", "blood sugar", "glucose", "bmi", "insulin",
        "pregnancies", "pedigree", "hba1c", "a1c",
    ])
    if diabetes:
        if any(p in text for p in [
            "what values", "what inputs", "what data", "values used",
            "inputs used", "what did i enter", "what information was used",
            "show my inputs", "show input values", "show values used",
        ]):
            return "diabetes", "inputs"
        if any(p in text for p in ["why", "explain my", "explain this", "what caused"]):
            return "diabetes", "why"
        if any(p in text for p in ["factors", "what affects", "risk factors"]):
            return "diabetes", "factors"
        if any(p in text for p in ["what is hba1c", "what does hba1c", "hba1c", "a1c"]):
            return "diabetes", "hba1c"
        if any(p in text for p in ["what is glucose", "what does glucose", "explain glucose"]):
            return "diabetes", "glucose"
        if any(p in text for p in ["what is bmi", "what does bmi", "explain bmi"]):
            return "diabetes", "bmi"
        return "diabetes", "latest"

    return None, None


def _format_disease_latest(latest: dict | None, label: str) -> str:
    if not latest:
        return f"I do not have a stored {label} assessment for your account yet. Complete the {label} assessment first, then I can explain its result."
    prediction = latest.get("prediction", "Not available")
    confidence = latest.get("confidence", "Not available")
    high = latest.get("high_risk_probability")
    date = latest.get("prediction_date", "Not available")
    lines = [
        f"Your latest {label} assessment result is **{prediction}**.",
        f"Predicted-class probability: {confidence}%.",
    ]
    if high is not None:
        lines.append(f"High-risk class probability: {float(high):.2f}%.")
    lines += [f"Assessment date: {date}.", "", "This is a machine-learning risk estimate, not a medical diagnosis."]
    return "\n".join(lines)


def _format_disease_inputs(latest: dict | None, fields: list[tuple[str, str, str]], label: str) -> str:
    if not latest:
        return f"I do not have a stored {label} assessment yet."
    data = latest.get("input_data") or {}
    if not data:
        return f"The latest {label} assessment exists, but its detailed input values are not available in the stored record."
    lines = [f"These are the values used in your latest {label} assessment:", ""]
    shown = 0
    for key, title, unit in fields:
        if key in data:
            value = data[key]
            suffix = f" {unit}" if unit else ""
            lines.append(f"• {title}: {value}{suffix}")
            shown += 1
    # Preserve any future/unknown fields instead of silently hiding them.
    known = {x[0] for x in fields}
    for key, value in data.items():
        if key not in known:
            lines.append(f"• {key.replace('_', ' ').title()}: {value}")
            shown += 1
    if shown == 0:
        return f"The latest {label} assessment exists, but no readable input values were stored."
    lines += ["", "These are the values provided to the machine-learning model; they are not, by themselves, a medical diagnosis."]
    return "\n".join(lines)


def _format_disease_why(latest: dict | None, label: str) -> str:
    if not latest:
        return f"I do not have a stored {label} assessment yet."
    return (
        f"Your latest {label} assessment was **{latest.get('prediction', 'Not available')}** "
        f"with a predicted-class probability of {latest.get('confidence', 'Not available')}%.\n\n"
        "The model considers the combined pattern of the input values rather than relying on one value alone.\n\n"
        + _format_disease_inputs(latest, {
            "heart": HEART_FIELDS,
            "diabetes": DIABETES_FIELDS_DISPLAY,
            "early_diabetes": EARLY_DIABETES_FIELDS,
        }[label if label in {"heart", "diabetes", "early_diabetes"} else "diabetes"], label)
        + "\n\nThis is a model-based risk estimate, not a medical diagnosis."
    )


def _diabetes_faq(intent: str) -> str:
    answers = {
        "glucose": "Glucose is a type of sugar used by the body for energy. The diabetes model uses glucose as one input feature. A single model input should not be interpreted as a diagnosis.",
        "bmi": "BMI is a height-and-weight-based measure used in the diabetes model as one input feature. It is only one part of the model input pattern and does not by itself diagnose diabetes.",
        "hba1c": "HbA1c (A1C) is a laboratory measurement commonly used by clinicians to assess average blood-glucose levels over roughly the previous 2–3 months. The current diabetes prediction form does not use HbA1c as one of its eight model inputs.",
        "factors": "The current diabetes model uses eight inputs: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age. The model evaluates their combined pattern rather than relying on one value alone.",
    }
    return answers.get(intent, "I can explain your diabetes assessment, its eight model inputs, glucose, BMI, HbA1c, predicted probability, or the difference between a model estimate and a clinical diagnosis.")


# ============================================================
# PREDICTION INTENT
# ============================================================

def _prediction_intent(message: str) -> str | None:
    """
    Detect application-specific prediction questions.

    Returns:

        latest
        history
        inputs
        why
        None
    """

    text = _normalise(message)

    # ========================================================
    # WHY DID I GET THIS PREDICTION?
    # ========================================================

    why_patterns = [

        "why did i get this prediction",
        "why did i get this result",
        "why was i given this prediction",
        "why was i given this result",
        "why was i predicted",

        "why this prediction",
        "why this result",

        "why am i at risk",
        "why is my risk high",
        "why is my risk low",

        "explain my prediction",
        "explain this prediction",
        "explain my result",
        "explain this result",

        "what caused my prediction",

        "why did the model predict",
        "why did the model give",

        "why is my prediction",
    ]


    # ========================================================
    # LATEST PREDICTION
    # ========================================================

    latest_patterns = [

        "what was my latest prediction",
        "what is my latest prediction",

        "my latest prediction",

        "show my latest prediction",

        "tell me my latest prediction",

        "what did i get in my prediction",

        "what was my result",
        "what is my result",

        "my latest result",

        "show my result",

        "latest prediction",
        "latest result",
    ]


    # ========================================================
    # INPUT VALUES
    # ========================================================

    input_patterns = [

        "what values did i enter",

        "what inputs did i enter",

        "what data was used",

        "what values were used",

        "inputs used for my prediction",

        "values used for my prediction",

        "show my input values",

        "show my inputs",

        "show input values",

        "show values used",

        "what information did i enter",

        "what information was used",

        "what did i enter",
    ]


    # ========================================================
    # PREDICTION HISTORY
    # ========================================================

    history_patterns = [

        "show my previous predictions",

        "show previous predictions",

        "show my prediction history",

        "show prediction history",

        "my prediction history",

        "prediction history",

        "previous predictions",

        "previous results",

        "show my previous results",

        "what were my previous predictions",

        "what were my previous results",

        "show all my predictions",

        "show all predictions",

        "list my predictions",

        "list previous predictions",

        "my past predictions",

        "show my past predictions",

        "what are my previous predictions",

        "what are my past predictions",

        "my assessment history",

        "show my assessment history",

        "previous assessments",

        "show previous assessments",

        "show my previous assessments",

        "past assessments",

        "show my past assessments",

        "how many predictions have i made",

        "how many assessments have i made",

        "how many predictions",

        "how many assessments",
    ]


    # ========================================================
    # INTENT CHECK
    # ========================================================

    if any(
        pattern in text
        for pattern in why_patterns
    ):
        return "why"


    # History is checked before latest because phrases such as
    # "previous results" should never go to Ollama.

    if any(
        pattern in text
        for pattern in history_patterns
    ):
        return "history"


    if any(
        pattern in text
        for pattern in latest_patterns
    ):
        return "latest"


    if any(
        pattern in text
        for pattern in input_patterns
    ):
        return "inputs"


    return None


# ============================================================
# FORMAT LATEST PREDICTION
# ============================================================

def _format_latest_prediction(
    latest: dict
) -> str:

    disease = latest.get(
        "disease_name",
        "Health"
    )

    prediction = latest.get(
        "prediction",
        "Not available"
    )

    confidence = latest.get(
        "confidence",
        "Not available"
    )
    high_risk = latest.get("high_risk_probability")

    prediction_date = latest.get(
        "prediction_date",
        "Not available"
    )

    return (
        f"Your latest {disease} assessment result is "
        f"**{prediction}**.\n\n"

        f"Predicted-class probability: {confidence}%.\n"
        + (f"High-risk class probability: {float(high_risk):.2f}%.\n" if high_risk is not None else "")
        + f"Assessment date: {prediction_date}.\n\n"

        "This is a machine-learning risk estimate, "
        "not a medical diagnosis."
    )


# ============================================================
# FORMAT INPUT VALUES
# ============================================================

def _format_inputs(
    latest: dict
) -> str:

    input_data = (
        latest.get("input_data")
        or {}
    )

    if not input_data:

        return (
            "The detailed input values are not available "
            "for your latest stored assessment."
        )


    lines = [
        "These are the values used for your latest assessment:",
        ""
    ]


    for field, value in input_data.items():

        label = FIELD_LABELS.get(
            field,
            field.replace("_", " ").title()
        )

        lines.append(
            f"• {label}: {value}"
        )


    lines.extend([
        "",
        (
            "These are the values provided to the "
            "machine-learning model. They should not "
            "be interpreted as a medical diagnosis."
        )
    ])


    return "\n".join(lines)


# ============================================================
# FORMAT PREDICTION EXPLANATION
# ============================================================

def _format_prediction_explanation(
    latest: dict
) -> str:

    disease = latest.get(
        "disease_name",
        "assessment"
    )

    prediction = latest.get(
        "prediction",
        "Not available"
    )

    confidence = latest.get(
        "confidence",
        "Not available"
    )

    input_data = (
        latest.get("input_data")
        or {}
    )


    lines = [

        (
            f"Your latest {disease} assessment was: "
            f"{prediction} "
            f"(predicted-class probability: {confidence}%)."
        ),

        "",

        "Why you received this result:",

        (
            "The machine-learning model uses the combination "
            "of the values entered for your assessment. "
            "It is not based on one value alone."
        ),
    ]


    if input_data:

        lines.append(
            "The values used for this assessment were:"
        )


        for field, value in input_data.items():

            label = FIELD_LABELS.get(
                field,
                field.replace("_", " ").title()
            )

            lines.append(
                f"• {label}: {value}"
            )


    else:

        lines.append(
            "The detailed input values for this assessment "
            "are not available in the stored prediction context."
        )


    lines.extend([

        "",

        (
            "These values are the inputs provided to the model; "
            "they should not be interpreted as proof that any "
            "one factor caused the result."
        ),

        (
            "The model considers their combined pattern."
        ),

        "",

        (
            "This is a model-based risk estimate, not a medical "
            "diagnosis. A qualified healthcare professional "
            "should interpret your health information in "
            "clinical context."
        ),
    ])


    return "\n".join(lines)


# ============================================================
# FORMAT PREDICTION HISTORY
# ============================================================

def _format_prediction_history(
    history: list,
    total_predictions: int
) -> str:
    """
    Format prediction history received from app.py.

    The records come directly from the existing
    `predictions` table.
    """

    if not history:

        return (
            "You do not have any previous predictions yet. "
            "Please complete a health assessment first."
        )


    lines = [

        "Your Prediction History",

        "",

        (
            f"You have completed {total_predictions} "
            f"health assessment"
            f"{'s' if total_predictions != 1 else ''}."
        ),

        ""
    ]


    for index, item in enumerate(
        history,
        start=1
    ):

        disease = item.get(
            "disease_name",
            "Health Assessment"
        )

        prediction = item.get(
            "prediction",
            "Not available"
        )

        confidence = item.get(
            "confidence"
        )
        high_risk = item.get("high_risk_probability")

        prediction_date = item.get(
            "prediction_date",
            "Date unavailable"
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if confidence is not None:

            try:

                confidence_text = (
                    f"{float(confidence):.2f}%"
                )

            except (
                TypeError,
                ValueError
            ):

                confidence_text = (
                    "Unavailable"
                )

        else:

            confidence_text = (
                "Unavailable"
            )


        # ----------------------------------------------------
        # HISTORY ITEM
        # ----------------------------------------------------

        lines.append(
            f"{index}. {disease} — {prediction}"
        )

        lines.append(
            f"   Predicted-class probability: {confidence_text}"
        )
        if high_risk is not None:
            try:
                lines.append(f"   High-risk class probability: {float(high_risk):.2f}%")
            except (TypeError, ValueError):
                pass

        lines.append(
            f"   Assessment date: {prediction_date}"
        )


        if index < len(history):

            lines.append("")


    lines.extend([

        "",

        (
            "These results are machine-learning risk "
            "estimates and not medical diagnoses."
        )
    ])


    return "\n".join(lines)


# ============================================================
# MAIN HEALTH ASSISTANT
# ============================================================

def ask_health_assistant(
    message: str,
    context: dict | None = None
) -> str:
    """
    Answer a health-related question.

    Application-specific prediction questions are handled
    locally using stored database context.

    General health questions are sent to Ollama.
    """

    message = (
        message or ""
    ).strip()


    if not message:

        return (
            "Please enter a health-related question."
        )


    context = context or {}


    latest = (
        context.get(
            "latest_prediction"
        )
        or {}
    )


    # ========================================================
    # DISEASE-SPECIFIC APPLICATION INTENT
    # ========================================================

    disease, disease_intent = _disease_request(message)
    if disease:
        context_key = {
            "heart": "latest_heart_prediction",
            "diabetes": "latest_diabetes_prediction",
            "early_diabetes": "latest_early_diabetes_prediction",
        }[disease]
        latest_disease = context.get(context_key) or {}
        label = {
            "heart": "heart disease",
            "diabetes": "diabetes",
            "early_diabetes": "early diabetes",
        }[disease]
        fields = {
            "heart": HEART_FIELDS,
            "diabetes": DIABETES_FIELDS_DISPLAY,
            "early_diabetes": EARLY_DIABETES_FIELDS,
        }[disease]
        if disease_intent == "inputs":
            return _format_disease_inputs(latest_disease, fields, label)
        if disease_intent == "why":
            return _format_disease_why(latest_disease, label)
        if disease_intent == "factors":
            if disease == "diabetes":
                return _diabetes_faq("factors")
            return (
                f"The {label} model evaluates the combined pattern of its stored input features. "
                "The individual inputs used in your latest assessment can be displayed with: "
                f"'What values were used in my latest {label} assessment?'"
            )
        if disease == "diabetes" and disease_intent in {"hba1c", "glucose", "bmi"}:
            return _diabetes_faq(disease_intent)
        return _format_disease_latest(latest_disease, label)

    intent = _prediction_intent(message)


    if intent:

        # ====================================================
        # PREDICTION HISTORY
        # ====================================================

        if intent == "history":

            history = (
                context.get(
                    "prediction_history"
                )
                or []
            )


            total_predictions = context.get(
                "total_predictions",
                len(history)
            )


            return _format_prediction_history(
                history,
                total_predictions
            )


        # ====================================================
        # NO LATEST PREDICTION
        # ====================================================

        if not latest:

            return (
                "I don't have a previous prediction "
                "available for your account. Please "
                "complete an assessment first."
            )


        # ====================================================
        # LATEST RESULT
        # ====================================================

        if intent == "latest":

            return _format_latest_prediction(
                latest
            )


        # ====================================================
        # INPUT VALUES
        # ====================================================

        if intent == "inputs":

            return _format_inputs(
                latest
            )


        # ====================================================
        # WHY / EXPLANATION
        # ====================================================

        if intent == "why":

            return _format_prediction_explanation(
                latest
            )


    # ========================================================
    # GENERAL HEALTH QUESTION
    # ========================================================

    latest_text = (
        "No previous prediction is available."
    )


    if latest:

        latest_text = (

            f"Disease/assessment type: "
            f"{latest.get('disease_name', 'Not available')}\n"

            f"Prediction result: "
            f"{latest.get('prediction', 'Not available')}\n"

            f"Confidence: "
            f"{latest.get('confidence', 'Not available')}\n"

            f"Assessment date: "
            f"{latest.get('prediction_date', 'Not available')}"
        )


    prompt = f"""{SYSTEM_PROMPT}

Application context:

{latest_text}

Total previous assessments:
{context.get('total_predictions', 0)}

Medical reports stored:
{context.get('medical_reports', 0)}

User question:
{message}

Answer directly in simple language.

If the question asks for a diagnosis or personalized treatment,
explain that the assistant cannot provide that and recommend a
qualified healthcare professional.

Do not change the stored application prediction.
Do not claim that a prediction is a medical diagnosis.
"""


    # ========================================================
    # OLLAMA REQUEST
    # ========================================================

    try:

        response = requests.post(

            OLLAMA_URL,

            json={

                "model":
                    OLLAMA_MODEL,

                "prompt":
                    prompt,

                "stream":
                    False,

                "options": {

                    "temperature":
                        0.2,

                    "num_predict":
                        180,
                },
            },

            timeout=30,
        )


        response.raise_for_status()


        data = response.json()


        answer = (
            data.get("response")
            or ""
        ).strip()


        if not answer:

            return (
                "I could not generate a response "
                "right now. Please try again."
            )


        return answer


    except requests.exceptions.ConnectionError:

        return (
            "The AI Health Assistant is temporarily "
            "unavailable. Please make sure Ollama "
            "is running and try again."
        )


    except requests.exceptions.Timeout:

        return (
            "The AI Health Assistant took too long "
            "to respond. Please try again."
        )


    except requests.exceptions.RequestException:

        return (
            "The AI Health Assistant is temporarily "
            "unavailable. Please make sure Ollama "
            "is running and try again."
        )


    except Exception as e:

        print(
            "OLLAMA HEALTH ASSISTANT ERROR:",
            e
        )

        return (
            "The AI Health Assistant is temporarily "
            "unavailable. Please try again."
        )