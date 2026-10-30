import re


# ============================================================
# REPORT TYPE DETECTOR
# ============================================================

def detect_report_type(text):

    if not text:
        return "general"

    text = text.lower()

    # --------------------------------------------------------
    # 1. STRONG REPORT TITLE / DOCUMENT IDENTIFIERS
    # --------------------------------------------------------
    # These should have the highest priority.
    # A report containing "Diabetes" as one parameter should
    # NOT automatically become a diabetes report.
    # --------------------------------------------------------

    # Heart / Cardiac report
    cardiac_title_patterns = [
        r"cardio\s+health\s+test\s+report",
        r"cardiac\s+health\s+report",
        r"cardiac\s+test\s+report",
        r"heart\s+health\s+report",
        r"heart\s+test\s+report",
        r"cardiology\s+report",
        r"cardiology\s+test",
        r"cardiac\s+profile"
    ]

    for pattern in cardiac_title_patterns:

        if re.search(pattern, text, re.IGNORECASE):
            return "cardiac"


    # Diabetes report
    diabetes_title_patterns = [
        r"diabetes\s+test\s+report",
        r"diabetes\s+health\s+report",
        r"diabetic\s+health\s+report",
        r"diabetes\s+screening\s+report",
        r"diabetes\s+profile",
        r"diabetic\s+profile"
    ]

    for pattern in diabetes_title_patterns:

        if re.search(pattern, text, re.IGNORECASE):
            return "diabetes"


    # Kidney report
    kidney_title_patterns = [
        r"kidney\s+test\s+report",
        r"kidney\s+health\s+report",
        r"renal\s+function\s+test",
        r"renal\s+function\s+report",
        r"kidney\s+profile",
        r"renal\s+profile"
    ]

    for pattern in kidney_title_patterns:

        if re.search(pattern, text, re.IGNORECASE):
            return "kidney"


    # CBC report
    cbc_title_patterns = [
        r"cbc\s+report",
        r"complete\s+blood\s+count",
        r"complete\s+blood\s+count\s+report",
        r"hematology\s+report",
        r"haematology\s+report"
    ]

    for pattern in cbc_title_patterns:

        if re.search(pattern, text, re.IGNORECASE):
            return "cbc"


    # --------------------------------------------------------
    # 2. STRONG CLINICAL COMBINATIONS
    # --------------------------------------------------------
    # Only classify based on clinical parameters when a clear
    # report title is not available.
    # --------------------------------------------------------

    # Cardiac indicators
    cardiac_score = 0

    cardiac_indicators = [
        "ejection fraction",
        "lvef",
        "left ventricular",
        "heart failure",
        "cardiac",
        "cardiology",
        "ecg",
        "electrocardiogram",
        "troponin",
        "bnp",
        "nt-probnp"
    ]

    for indicator in cardiac_indicators:

        if indicator in text:
            cardiac_score += 1


    # Diabetes indicators
    diabetes_score = 0

    diabetes_indicators = [
        "hba1c",
        "fasting blood glucose",
        "fasting glucose",
        "blood glucose",
        "random blood glucose",
        "postprandial glucose",
        "insulin",
        "diabetes mellitus"
    ]

    for indicator in diabetes_indicators:

        if indicator in text:
            diabetes_score += 1


    # Kidney indicators
    kidney_score = 0

    kidney_indicators = [
        "serum creatinine",
        "blood urea",
        "blood urea nitrogen",
        "bun",
        "egfr",
        "glomerular filtration",
        "renal",
        "kidney"
    ]

    for indicator in kidney_indicators:

        if indicator in text:
            kidney_score += 1


    # CBC indicators
    cbc_score = 0

    cbc_indicators = [
        "hemoglobin",
        "haemoglobin",
        "white blood cell",
        "wbc",
        "red blood cell",
        "rbc",
        "platelet count",
        "hematocrit",
        "haematocrit"
    ]

    for indicator in cbc_indicators:

        if indicator in text:
            cbc_score += 1


    # --------------------------------------------------------
    # 3. SPECIAL CASE:
    # Diabetes alone should NOT classify a report as diabetes.
    #
    # For example:
    #
    # Diabetes: No
    #
    # should remain a general/cardio report depending on
    # other evidence.
    # --------------------------------------------------------

    scores = {
        "cardiac": cardiac_score,
        "diabetes": diabetes_score,
        "kidney": kidney_score,
        "cbc": cbc_score
    }

    highest_type = max(
        scores,
        key=scores.get
    )

    highest_score = scores[highest_type]

    # Require at least 2 strong indicators when there is
    # no explicit report title.

    if highest_score >= 2:
        return highest_type


    # --------------------------------------------------------
    # 4. GENERAL REPORT
    # --------------------------------------------------------

    return "general"