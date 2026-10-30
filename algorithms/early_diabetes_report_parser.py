# ============================================================
# EARLY DIABETES MEDICAL REPORT PARSER
# ============================================================

import re


EARLY_DIABETES_FIELDS = [
    "age",
    "gender",
    "polyuria",
    "polydipsia",
    "sudden_weight_loss",
    "weakness",
    "polyphagia",
    "genital_thrush",
    "visual_blurring",
    "itching",
    "irritability",
    "delayed_healing",
    "partial_paresis",
    "muscle_stiffness",
    "alopecia",
    "obesity",
]


# ------------------------------------------------------------
# LABELS / SYNONYMS
# ------------------------------------------------------------

FIELD_LABELS = {

    "age": [
        "age",
        "patient age",
        "age (years)",
    ],

    "gender": [
        "gender",
        "sex",
        "patient gender",
    ],

    "polyuria": [
        "polyuria",
        "excessive urination",
        "frequent urination",
        "increased urination",
    ],

    "polydipsia": [
        "polydipsia",
        "excessive thirst",
        "increased thirst",
        "frequent thirst",
    ],

    "sudden_weight_loss": [
        "sudden weight loss",
        "sudden weight-loss",
        "unexplained weight loss",
        "weight loss",
    ],

    "weakness": [
        "weakness",
        "general weakness",
        "fatigue",
    ],

    "polyphagia": [
        "polyphagia",
        "excessive hunger",
        "increased hunger",
        "increased appetite",
    ],

    "genital_thrush": [
        "genital thrush",
        "genital yeast infection",
        "yeast infection",
    ],

    "visual_blurring": [
        "visual blurring",
        "blurred vision",
        "blurry vision",
        "vision problems",
    ],

    "itching": [
        "itching",
        "itchiness",
    ],

    "irritability": [
        "irritability",
        "irritable",
    ],

    "delayed_healing": [
        "delayed healing",
        "slow healing",
        "poor wound healing",
    ],

    "partial_paresis": [
        "partial paresis",
        "partial weakness",
        "limb weakness",
    ],

    "muscle_stiffness": [
        "muscle stiffness",
        "stiffness",
    ],

    "alopecia": [
        "alopecia",
        "hair loss",
    ],

    "obesity": [
        "obesity",
        "obese",
    ],
}


# ------------------------------------------------------------
# NORMALIZATION
# ------------------------------------------------------------

def normalize_yes_no(value):

    if value is None:
        return None

    value = str(value).strip().lower()

    if value in {
        "yes",
        "y",
        "true",
        "positive",
        "present",
        "1",
        "positive (+)",
    }:
        return "Yes"

    if value in {
        "no",
        "n",
        "false",
        "negative",
        "absent",
        "0",
        "negative (-)",
    }:
        return "No"

    return None


# ------------------------------------------------------------
# SEARCH VALUE AFTER LABEL
# ------------------------------------------------------------

def _extract_after_label(text, labels, max_chars=80):

    for label in sorted(labels, key=len, reverse=True):

        escaped = re.escape(label)

        pattern = rf"""
            (?<![A-Za-z])
            {escaped}
            \s*
            (?:
                :|-|–|—|=|\|
            )?
            \s*
            ([^\n\r]{{1,{max_chars}}})
        """

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.VERBOSE
        )

        if match:

            value = match.group(1).strip()

            # Remove trailing common report separators
            value = re.split(
                r"\s{2,}|\t|\|",
                value
            )[0].strip()

            return value

    return None


# ------------------------------------------------------------
# AGE
# ------------------------------------------------------------

def _extract_age(text):

    value = _extract_after_label(
        text,
        FIELD_LABELS["age"],
        max_chars=20
    )

    if value:

        match = re.search(
            r"\b(\d{1,3})\b",
            value
        )

        if match:

            age = int(match.group(1))

            if 1 <= age <= 120:
                return age

    return None


# ------------------------------------------------------------
# GENDER
# ------------------------------------------------------------

def _extract_gender(text):

    value = _extract_after_label(
        text,
        FIELD_LABELS["gender"],
        max_chars=30
    )

    if not value:
        return None

    value_lower = value.lower()

    if re.search(r"\bfemale\b|\bf\b", value_lower):
        return "Female"

    if re.search(r"\bmale\b|\bm\b", value_lower):
        return "Male"

    return None


# ------------------------------------------------------------
# YES / NO FIELD
# ------------------------------------------------------------

def _extract_yes_no_field(text, field):

    labels = FIELD_LABELS[field]

    value = _extract_after_label(
        text,
        labels,
        max_chars=50
    )

    if value:

        normalized = normalize_yes_no(value)

        if normalized:
            return normalized

        # Explicit first word
        first_word = value.split()[0]

        normalized = normalize_yes_no(first_word)

        if normalized:
            return normalized

    return None


# ------------------------------------------------------------
# MAIN PARSER
# ------------------------------------------------------------

def parse_early_diabetes_report(text):

    if not text:
        return {}

    # Normalize PDF text
    text = str(text)

    text = text.replace(
        "\u00a0",
        " "
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    extracted = {}

    # Age
    age = _extract_age(text)

    if age is not None:
        extracted["age"] = age

    # Gender
    gender = _extract_gender(text)

    if gender is not None:
        extracted["gender"] = gender

    # Symptoms
    symptom_fields = [
        "polyuria",
        "polydipsia",
        "sudden_weight_loss",
        "weakness",
        "polyphagia",
        "genital_thrush",
        "visual_blurring",
        "itching",
        "irritability",
        "delayed_healing",
        "partial_paresis",
        "muscle_stiffness",
        "alopecia",
        "obesity",
    ]

    for field in symptom_fields:

        value = _extract_yes_no_field(
            text,
            field
        )

        if value is not None:
            extracted[field] = value

    return extracted


# ------------------------------------------------------------
# ENSURE ALL FIELDS
# ------------------------------------------------------------

def ensure_early_diabetes_fields(data):

    result = {}

    for field in EARLY_DIABETES_FIELDS:

        result[field] = data.get(field)

    return result


# ------------------------------------------------------------
# COUNTS
# ------------------------------------------------------------

def count_early_diabetes_fields(data):

    extracted_count = sum(
        1
        for field in EARLY_DIABETES_FIELDS
        if data.get(field) not in (None, "")
    )

    missing_count = (
        len(EARLY_DIABETES_FIELDS)
        - extracted_count
    )

    return extracted_count, missing_count