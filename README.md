# 🩺 Explainable Multi-Disease Prediction & Personalized Health Intelligence System

An AI-powered web application for **multi-disease risk assessment, explainable machine learning, longitudinal health-risk tracking, and personalized preventive health intelligence**.

The system currently supports:

- ❤️ Heart Disease Risk Assessment
- 🩸 Diabetes Risk Assessment
- 🔎 Early Diabetes Risk Assessment

It combines machine-learning models, adaptive ensemble learning, Explainable AI (XAI), medical-report processing, longitudinal analytics, and personalized educational recommendations within a Flask-based healthcare application.

> **Important:** This project is intended for educational and research purposes. Model outputs represent machine-learning risk estimates and are not medical diagnoses or substitutes for professional healthcare advice.

---

## 📌 Project Overview

Traditional machine-learning disease prediction applications often provide only a final prediction such as **High Risk** or **Low Risk**, without explaining why the model produced that result or how the user's assessments change over time.

This project extends conventional disease prediction by integrating:

- Multi-disease machine-learning models
- Adaptive ensemble learning
- Explainable AI using SHAP
- Model uncertainty and disagreement analysis
- Longitudinal risk tracking
- What-If risk simulation
- Cross-disease health intelligence
- Personalized risk-reduction recommendations
- Medical-report data extraction
- Prediction audit and model versioning
- AI-assisted health information
- Interactive analytics dashboards

The objective is to provide a more **transparent, research-oriented, and user-friendly health-risk assessment platform**.

---

# ✨ Key Features

## ❤️ 1. Heart Disease Risk Assessment

Predicts model-estimated heart-disease risk using clinical and health-related parameters including:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

The heart module uses multiple machine-learning algorithms together with an adaptive ensemble approach.

---

## 🩸 2. Diabetes Risk Assessment

Provides diabetes risk assessment using parameters such as:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The module also provides risk-factor analysis, explanations, educational recommendations, and assessment history.

---

## 🔎 3. Early Diabetes Risk Assessment

A separate symptom-oriented model is included for early diabetes risk analysis.

Inputs include factors such as:

- Age
- Gender
- Polyuria
- Polydipsia
- Sudden Weight Loss
- Weakness
- Polyphagia
- Genital Thrush
- Visual Blurring
- Itching
- Irritability
- Delayed Healing
- Partial Paresis
- Muscle Stiffness
- Alopecia
- Obesity

This model is maintained separately from the standard diabetes assessment.

---

# 🧠 Adaptive Ensemble Learning

The heart-disease prediction module combines multiple machine-learning algorithms, including:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

Instead of relying entirely on a single classifier, the system uses an **adaptive ensemble strategy** that combines predictions from multiple models.

Adaptive weighting considers model-level performance together with prediction confidence to produce the final ensemble output.

---

# 🔍 Explainable AI — SHAP

The system integrates **SHAP-based Explainable AI (XAI)** to improve prediction transparency.

Instead of showing only:

> High Risk

the application can also help explain which input features influenced the machine-learning output.

This allows users and researchers to better understand the model's behaviour.

---

# 📈 Longitudinal Risk Trend Engine

The application stores previous assessments and tracks model-estimated high-risk probability over time.

Users can view:

- Previous assessments
- Latest assessment
- Probability trends
- Change from the previous assessment
- Increasing/decreasing/stable model-output patterns

The trend represents changes in **model output across assessments** and should not be interpreted as clinically validated disease progression.

---

# 🧪 What-If Risk Simulator

The What-If Simulator allows users to modify selected input values and observe how the trained model responds.

Example:

```text
Saved Assessment
Heart high-risk probability: 68%

Hypothetical Scenario
Resting BP: 148 → 130
Cholesterol: 250 → 210

Simulated model probability: 44%

Difference: -24 percentage points
```

What-If simulations are kept separate from genuine prediction history.

> The simulator demonstrates model behaviour under hypothetical inputs. It does not predict the medical benefit of a treatment or lifestyle intervention.

---

# ⚖️ Model Uncertainty & Disagreement Analysis

For ensemble-based assessments, the system can expose individual model outputs and adaptive weights.

This allows analysis of:

- Individual model probabilities
- Adaptive model weights
- Probability spread
- Model agreement/disagreement
- Prediction uncertainty

This improves transparency compared with presenting only a single ensemble probability.

---

# 🔗 Cross-Disease Health Intelligence

A unified intelligence layer combines the latest available information from:

- Heart Disease
- Diabetes
- Early Diabetes

The module provides:

- Disease-specific latest predictions
- Disease-specific longitudinal trends
- Shared recorded attention signals
- Assessment counts
- Cross-disease overview

The application does **not** treat probabilities from different disease models as directly equivalent clinical risks.

Any descriptive combined index is presented only as a research-oriented summary and not as a medical risk score.

---

# 📄 Medical Report Processing

The system supports medical-report upload and structured data extraction.

The workflow can:

1. Upload a medical report
2. Extract available supported parameters
3. Normalize detected values
4. Identify missing parameters
5. Pre-fill compatible assessment fields
6. Allow the user to verify or complete missing information
7. Continue with disease-risk assessment

This reduces repetitive manual data entry while keeping user verification part of the workflow.

---

# 💡 Personalized Risk-Reduction Recommendations

After an assessment, the system generates educational recommendations based on:

- Model output
- Recorded risk-related inputs
- Rule-based attention signals
- Explainability information

The recommendation layer is designed for **preventive-health education** and does not prescribe medication or replace professional medical advice.

---

# 📊 Prediction History Center

The system stores previous assessments in MySQL and provides structured prediction history.

History functionality includes:

- Disease-specific records
- Prediction result
- High-risk probability
- Assessment date
- Historical report access
- Longitudinal trend analysis

The multi-disease architecture supports separate tracking of Heart Disease, Diabetes, and Early Diabetes assessments.

---

# 🧾 Prediction Audit & Model Versioning

Prediction records can be associated with audit information such as:

- Prediction ID
- Disease module
- Model name
- Model version
- Model artifact
- Dataset reference
- Decision threshold
- Model probability
- Prediction result
- Input snapshot
- Timestamp

This improves **traceability and reproducibility** of machine-learning assessments.

---

# 🤖 AI Health Assistant

The application contains an AI-assisted health-information interface capable of answering supported questions about:

- Latest prediction
- Values used in an assessment
- Previous disease assessments
- Model-related risk factors
- General educational health information

The assistant is designed with safety limitations and does not provide medical diagnosis or prescriptions.

---

# 🏥 Healthcare Facility Locator

The healthcare locator helps users discover nearby healthcare facilities such as:

- Hospitals
- Doctors/clinics
- Pharmacies
- Diagnostic laboratories

The module supports location-based searching, map visualization, distance information, and external navigation/directions.

---

# 🔔 Notification Center

The application includes a notification system for relevant system and assessment events.

Features include:

- Read/unread status
- Notification categories
- Priority indicators
- Search/filtering
- Links to relevant assessments or reports

---

# 🔐 Authentication & Security

The application includes:

- User registration
- Secure login
- Password hashing using bcrypt
- Flask session management
- Forgot-password workflow
- Email OTP verification
- OTP expiration
- Limited OTP attempts
- One-time OTP usage
- Secure password reset

---

# 📊 Research & Model Evaluation

The project contains research-oriented model evaluation using metrics such as:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Brier Score
- Confusion Matrix
- Cross-validation stability

Repeated cross-validation is used to compare ensemble strategies more reliably than relying on a single train/test split.

The project intentionally avoids claiming that adaptive weighting always outperforms static ensemble methods; results are interpreted according to the measured experimental performance.

---

# 🛠️ Technology Stack

### Backend

- Python
- Flask
- Flask-MySQLdb
- Flask-Bcrypt

### Machine Learning

- Scikit-learn
- Pandas
- NumPy
- SHAP
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine
- Adaptive Ensemble Learning

### Database

- MySQL
- XAMPP

### Frontend

- HTML5
- CSS3
- Bootstrap
- JavaScript
- Jinja2
- Chart.js

### Additional Technologies

- REST/API integration
- SMTP Email Integration
- PDF/Medical Report Processing
- Leaflet Maps
- OpenStreetMap-based location services
- Ollama-based local AI integration

---

# 🏗️ System Architecture

```text
                     USER
                       │
                       ▼
                Flask Web Application
                       │
       ┌───────────────┼────────────────┐
       │               │                │
       ▼               ▼                ▼
 Authentication   Disease Models   Report Processing
       │               │                │
       │       ┌───────┼────────┐       │
       │       ▼       ▼        ▼       │
       │     Heart  Diabetes  Early     │
       │                    Diabetes    │
       │               │                │
       │               ▼                │
       │        Prediction Engine       │
       │               │                │
       │       ┌───────┼────────┐       │
       │       ▼       ▼        ▼       │
       │      XAI   Uncertainty What-If │
       │               │                │
       └───────────────┼────────────────┘
                       │
                       ▼
              Health Intelligence
                       │
           ┌───────────┼───────────┐
           ▼           ▼           ▼
        History      Trends    Recommendations
           │           │           │
           └───────────┼───────────┘
                       ▼
                     MySQL
```

---

# 📂 Project Structure

```text
MultiDiseasePrediction/
│
├── app.py
├── config.py
├── database_setup.py
│
├── algorithms/
│   ├── adaptive_ensemble.py
│   ├── diabetes_engine.py
│   ├── early_diabetes_engine.py
│   ├── recommendation.py
│   ├── research_validation.py
│   ├── xai_explainer.py
│   ├── report_parser.py
│   ├── report_normalizer.py
│   └── validation.py
│
├── ai/
│   └── health_assistant.py
│
├── models/
│   ├── heart/
│   ├── diabetes/
│   └── early_diabetes/
│
├── dataset/
│
├── templates/
│
├── static/
│
└── utils/
```



---

# 🚀 Installation & Setup

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd MultiDiseasePrediction
```

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure MySQL

Start **Apache and MySQL** using XAMPP.

Create/configure the project database according to `config.py`.

Run the database setup script if required:

```bash
python database_setup.py
```

## 5. Configure Email OTP

For password-reset email functionality, configure the required mail environment variables instead of committing credentials to GitHub.

Example:

```text
MAIL_USERNAME=your-email
MAIL_PASSWORD=your-app-password
MAIL_FROM=your-email
```



## 6. Optional — Start Ollama

If using the local AI Health Assistant, ensure Ollama and the configured local model are available.

## 7. Run the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 🔬 Research Contribution

The project explores whether a conventional disease-risk prediction system can be enhanced by combining:

1. Adaptive ensemble learning
2. Explainable AI
3. Model disagreement analysis
4. Longitudinal model-output tracking
5. Hypothetical What-If simulation
6. Personalized preventive-health information
7. Cross-disease intelligence
8. Prediction auditing and reproducibility

Rather than focusing only on classification accuracy, the project emphasizes **transparency, interpretability, longitudinal context, and responsible presentation of machine-learning predictions**.

---

# 🌍 Sustainable Development Goal Alignment

The project primarily aligns with:

### SDG 3 — Good Health and Well-Being

It explores the use of AI and data-driven systems for preventive-health awareness, early risk assessment, and accessible health information.

It also has supporting relevance to:

### SDG 9 — Industry, Innovation and Infrastructure

through the application of machine learning, explainable AI, and digital-health technologies.

---

# ⚠️ Disclaimer

This application is an **academic and research project**.

Predictions, probabilities, explanations, recommendations, simulations, and AI-generated information are intended for educational and research purposes only.

They:

- Do not constitute a medical diagnosis
- Do not confirm the presence or absence of disease
- Do not prescribe treatment or medication
- Should not be used as a substitute for professional medical evaluation

Users should consult qualified healthcare professionals for medical concerns.

---

# 👩‍💻 Author

**Malavika Manoj**  
Master of Computer Applications (MCA)

---

## ⭐ Support

If you find this academic project useful, consider giving the repository a ⭐.

Contributions, suggestions, and constructive feedback are welcome.
