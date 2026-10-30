# ============================================================
# HEART DISEASE PREDICTION SYSTEM
# Complete Flask Application
# ============================================================

from algorithms.report_normalizer import normalize_extracted_data
from algorithms.validation import validate_extracted_data
from algorithms.recommendation import generate_recommendations, generate_personalized_plan
from algorithms.model_compatibility import check_heart_model_compatibility
from algorithms.xai_explainer import explain_prediction
from algorithms.early_diabetes_report_parser import (
    parse_early_diabetes_report,
    ensure_early_diabetes_fields,
    count_early_diabetes_fields,
    EARLY_DIABETES_FIELDS
)



import joblib
import pandas as pd
import requests
import os
import json
import sys
import subprocess
import secrets
import hashlib
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
    send_from_directory
)

from flask_mysqldb import MySQL
from flask_bcrypt import Bcrypt
from werkzeug.utils import secure_filename

import config
from database_setup import ensure_database_and_tables

from utils.pdf_reader import extract_text

from algorithms.report_parser import parse_report
from algorithms.report_type_detector import detect_report_type
from ai.health_assistant import ask_health_assistant

# Diabetes module is additive: it does not replace or alter the Heart Disease model.
from algorithms.diabetes_engine import (
    train_diabetes_engine, DIABETES_FIELDS, DIABETES_COLUMNS
)
from algorithms.early_diabetes_engine import (
    train_early_diabetes_engine, EARLY_FIELDS
)
from algorithms.diabetes.report_parser import parse_diabetes_report


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

app.config["SECRET_KEY"] = config.SECRET_KEY

app.config["MYSQL_HOST"] = config.MYSQL_HOST
app.config["MYSQL_USER"] = config.MYSQL_USER
app.config["MYSQL_PASSWORD"] = config.MYSQL_PASSWORD
app.config["MYSQL_DB"] = config.MYSQL_DB
app.config["MYSQL_PORT"] = config.MYSQL_PORT
app.config["MYSQL_CONNECT_TIMEOUT"] = config.MYSQL_CONNECT_TIMEOUT

app.config["UPLOAD_FOLDER"] = "uploads"

# Maximum upload size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# INITIALIZE NEW DATABASE BEFORE FLASK-MYSQLDB
# ============================================================

# Creates only heart_prediction_new_db and its tables.
# The previous heart_prediction_db is never selected, altered, or migrated.
try:
    ensure_database_and_tables()
except Exception as e:
    print("=" * 70)
    print("NEW DATABASE INITIALIZATION FAILED")
    print(e)
    print("=" * 70)


# ============================================================
# INITIALIZE EXTENSIONS
# ============================================================

mysql = MySQL(app)
bcrypt = Bcrypt(app)


# ============================================================
# CREATE UPLOAD DIRECTORY
# ============================================================

os.makedirs(
    app.config["UPLOAD_FOLDER"],
    exist_ok=True
)


# ============================================================
# LOAD HEART DISEASE MODEL
# ============================================================

MODEL_PATH = "models/heart/ensemble_model.pkl"

try:

    heart_model = joblib.load(MODEL_PATH)

    print("=" * 70)
    print("HEART DISEASE MODEL LOADED SUCCESSFULLY")
    print("Model:", MODEL_PATH)
    print("=" * 70)

except Exception as e:

    print("=" * 70)
    print("ERROR LOADING HEART DISEASE MODEL")
    print("=" * 70)
    print(e)
    print("=" * 70)

    # A scikit-learn pickle can fail when it was created with a different
    # scikit-learn version (for example 1.8.x -> 1.9.x). Since this project
    # contains the full 918-row training dataset and training script, rebuild
    # the production model with the currently installed local environment.
    heart_model = None
    try:
        print("RETRAINING HEART MODEL WITH CURRENT PYTHON/SCIKIT-LEARN ENVIRONMENT...")
        subprocess.run(
            [sys.executable, "-m", "algorithms.train_new_heart_model"],
            check=True
        )
        heart_model = joblib.load(MODEL_PATH)
        print("HEART DISEASE MODEL RETRAINED AND LOADED SUCCESSFULLY")
    except Exception as retrain_error:
        print("AUTOMATIC MODEL REBUILD FAILED:", retrain_error)
        heart_model = None


# ============================================================
# DIABETES MODEL
# ============================================================

diabetes_engine = None
try:
    diabetes_engine = train_diabetes_engine(Path(BASE_DIR))
    print("=" * 70)
    print("DIABETES MODEL LOADED/TRAINED SUCCESSFULLY")
    print("Models:", ", ".join(diabetes_engine.models.keys()))
    print("=" * 70)
except Exception as diabetes_error:
    print("DIABETES MODEL INITIALIZATION FAILED:", diabetes_error)
    diabetes_engine = None


# ============================================================
# EARLY DIABETES MODEL (RESTORED FROM ORIGINAL DIABETES MODULE)
# ============================================================

early_diabetes_engine = None
try:
    early_diabetes_engine = train_early_diabetes_engine(Path(BASE_DIR))
    print("EARLY DIABETES MODEL LOADED/TRAINED SUCCESSFULLY")
    print("Models:", ", ".join(early_diabetes_engine.models.keys()))
except Exception as early_diabetes_error:
    print("EARLY DIABETES MODEL INITIALIZATION FAILED:", early_diabetes_error)
    early_diabetes_engine = None


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "qwen2.5:0.5b"


# ============================================================
# HEART MODEL FIELDS
# ============================================================

EXPECTED_FIELDS = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_bs", "resting_ecg", "max_hr", "exercise_angina",
    "oldpeak", "st_slope"
]

MODEL_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
    "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
    "Oldpeak", "ST_Slope"
]


# ============================================================
# LOGIN CHECK HELPER
# ============================================================

def login_required():

    return "user_id" in session

# ============================================================
# PROBABILITY INTERPRETATION HELPERS
# ============================================================

def high_risk_probability_from_prediction(prediction, predicted_class_probability):
    """Return probability assigned specifically to HeartDisease=1 (high-risk)."""
    try:
        p = max(0.0, min(100.0, float(predicted_class_probability or 0)))
    except (TypeError, ValueError):
        p = 0.0
    return round(p if "high" in str(prediction or "").lower() else 100.0 - p, 2)


def row_high_risk_probability(prediction, confidence, stored_high_risk=None):
    """Use explicit stored high-risk probability, with a legacy-row fallback."""
    if stored_high_risk is not None:
        try:
            return round(float(stored_high_risk), 2)
        except (TypeError, ValueError):
            pass
    return high_risk_probability_from_prediction(prediction, confidence)


# ============================================================
# NOTIFICATION HELPER
# ============================================================

def create_notification(
    user_id, title, message, notification_type="info",
    reference_type=None, reference_id=None, priority=None
):
    """Create an in-app notification with optional navigation metadata."""
    if priority is None:
        priority = {"danger": "important", "warning": "attention"}.get(notification_type, "normal")
    cur = None
    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            INSERT INTO notifications
            (user_id, title, message, notification_type, is_read, reference_type, reference_id, priority)
            VALUES (%s, %s, %s, %s, 0, %s, %s, %s)
        """, (user_id, title, message, notification_type, reference_type, reference_id, priority))
        mysql.connection.commit()
        return True
    except Exception as e:
        try: mysql.connection.rollback()
        except Exception: pass
        print("NOTIFICATION ERROR:", e)
        return False
    finally:
        if cur: cur.close()


# ============================================================
# UNREAD NOTIFICATION COUNT
# ============================================================

@app.context_processor
def inject_notification_count():

    if "user_id" not in session:

        return {
            "unread_notifications": 0
        }

    try:

        cur = mysql.connection.cursor()

        cur.execute(
            """
            SELECT COUNT(*)
            FROM notifications
            WHERE user_id = %s
            AND is_read = 0
            """,
            (session["user_id"],)
        )

        unread_count = cur.fetchone()[0]

        cur.close()

        return {
            "unread_notifications": unread_count
        }

    except Exception as e:

        print(
            "NOTIFICATION COUNT ERROR:",
            e
        )

        return {
            "unread_notifications": 0
        }

# ============================================================
# AI EXPLANATION
# ============================================================

def generate_ai_explanation(
    user_data,
    prediction,
    result,
    confidence,
    risk_factors
):
    """Generate an instant, deterministic explanation for the result page.

    The prediction page must not wait for Ollama. The AI Health Assistant
    remains available separately for conversational explanations.
    """

    prediction_text = str(result)
    confidence_text = f"{float(confidence):.1f}%" if confidence is not None else "not available"

    if risk_factors:
        factor_lines = "\n".join(f"- {factor}" for factor in risk_factors[:6])
    else:
        factor_lines = "- No major risk factors were flagged by the application's rule-based review."

    suggestions = [
        "Maintain a balanced diet and limit excess salt, sugar, and highly processed foods.",
        "Stay physically active at an appropriate level and avoid smoking.",
        "Discuss the result and any abnormal measurements with a qualified healthcare professional."
    ]

    return (
        "PREDICTION SUMMARY\n\n"
        f"The machine-learning model produced a {prediction_text.lower()} classification "
        f"with a predicted class probability of {confidence_text}. "
        "This result is a model-based risk estimate and is not a medical diagnosis.\n\n"
        "IMPORTANT RISK FACTORS\n\n"
        f"{factor_lines}\n\n"
        "PERSONALIZED RISK REDUCTION SUGGESTIONS\n\n"
        f"1. {suggestions[0]}\n"
        f"2. {suggestions[1]}\n"
        f"3. {suggestions[2]}\n\n"
        "WHEN TO CONSULT A DOCTOR\n\n"
        "Discuss this prediction and your clinical measurements with a qualified "
        "healthcare professional, especially if you have symptoms or concerns."
    )



# ============================================================
# RISK FACTOR GENERATION
# ============================================================

def generate_risk_factors(user_data):
    factors=[]
    if user_data["age"] >= 60: factors.append(f"Age ({int(user_data['age'])} years) contributed toward the model's higher-risk class in this assessment.")
    if user_data["resting_bp"] >= 140: factors.append(f"Resting blood pressure ({user_data['resting_bp']}) is elevated.")
    if user_data["cholesterol"] >= 240: factors.append(f"Cholesterol ({user_data['cholesterol']}) is high relative to commonly used categories.")
    if user_data["fasting_bs"] == 1: factors.append("Fasting blood sugar is at or above the dataset threshold (120 mg/dL equivalent encoding).")
    if user_data["exercise_angina"] == "Y": factors.append("Exercise-induced angina is recorded as present.")
    if user_data["oldpeak"] > 2: factors.append(f"Oldpeak ({user_data['oldpeak']}) is elevated in the recorded exercise ECG information.")
    if user_data["st_slope"] == "Flat": factors.append("ST slope is recorded as Flat.")
    elif user_data["st_slope"] == "Down": factors.append("ST slope is recorded as Down.")
    if user_data["chest_pain_type"] == "ASY": factors.append("Chest pain type is recorded as asymptomatic (ASY) in the dataset coding.")
    return factors

# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ABOUT
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if request.method == "POST":

        fullname = request.form.get(
            "fullname",
            ""
        ).strip()

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        if not fullname or not email or not password:

            flash(
                "Please fill in all required fields.",
                "warning"
            )

            return redirect(
                url_for("register")
            )


        try:

            cur = mysql.connection.cursor()


            # Check email

            cur.execute(
                """
                SELECT *
                FROM users
                WHERE email = %s
                """,
                (email,)
            )

            existing_user = cur.fetchone()


            if existing_user:

                cur.close()

                flash(
                    "Email already registered!",
                    "danger"
                )

                return redirect(
                    url_for("register")
                )


            # Hash password

            hashed_password = (
                bcrypt
                .generate_password_hash(password)
                .decode("utf-8")
            )


            # Insert user

            cur.execute(

                """
                INSERT INTO users
                (fullname, email, password)
                VALUES (%s, %s, %s)
                """,

                (
                    fullname,
                    email,
                    hashed_password
                )

            )


            mysql.connection.commit()

            cur.close()


            flash(
                "Registration successful! Please login.",
                "success"
            )

            return redirect(
                url_for("login")
            )


        except Exception as e:

            mysql.connection.rollback()

            print(
                "REGISTRATION ERROR:",
                e
            )

            flash(
                "Registration failed. Please try again.",
                "danger"
            )


    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )


        try:

            cur = mysql.connection.cursor()


            cur.execute(

                """
                SELECT *
                FROM users
                WHERE email = %s
                """,

                (email,)

            )

            user = cur.fetchone()

            cur.close()


            if user:

                if bcrypt.check_password_hash(
                    user[3],
                    password
                ):

                    session["logged_in"] = True

                    session["user_id"] = user[0]

                    session["fullname"] = user[1]

                    session["email"] = user[2]


                    flash(
                        "Login successful!",
                        "success"
                    )


                    return redirect(
                        url_for("dashboard")
                    )


            flash(
                "Invalid Email or Password!",
                "danger"
            )


        except Exception as e:

            print(
                "LOGIN ERROR:",
                e
            )

            flash(
                "Unable to login. Please try again.",
                "danger"
            )


    return render_template(
        "login.html"
    )




# ============================================================
# FORGOT PASSWORD / EMAIL OTP
# ============================================================

RESET_OTP_EXPIRY_MINUTES = 10
RESET_MAX_ATTEMPTS = 5
RESET_RESEND_SECONDS = 60


def generate_reset_otp():
    """Generate a cryptographically secure 6-digit OTP."""
    return f"{secrets.randbelow(1000000):06d}"


def hash_reset_otp(otp):
    """Store only a salted-by-project-secret hash of the OTP."""
    value = f"{config.SECRET_KEY}:{otp}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def send_password_reset_email(recipient_email, recipient_name, otp):
    """Send the password-reset OTP to the registered email address."""
    username = str(getattr(config, "MAIL_USERNAME", "") or "").strip()
    password = str(getattr(config, "MAIL_PASSWORD", "") or "").strip()
    mail_from = str(getattr(config, "MAIL_FROM", username) or username).strip()

    if not username or not password:
        raise RuntimeError("SMTP credentials are not configured in config.py")

    recipient_email = str(recipient_email or "").strip()
    if not recipient_email:
        raise ValueError("Recipient email address is empty")

    message = EmailMessage()
    message["From"] = mail_from
    message["To"] = recipient_email
    message["Reply-To"] = mail_from
    message["Subject"] = "AI Health Risk Assessment - Password Reset OTP"

    safe_name = str(recipient_name or "User").strip() or "User"
    body = f"""Hello {safe_name},

We received a request to reset your password for AI Health Risk Assessment.

Your password-reset OTP is: {otp}

This OTP is valid for {RESET_OTP_EXPIRY_MINUTES} minutes and can be used only once.
If you did not request a password reset, you can safely ignore this email.

Regards,
AI Health Risk Assessment
"""
    message.set_content(body)

    print("=" * 70)
    print("PASSWORD RESET EMAIL")
    print("SMTP server        :", getattr(config, "MAIL_SERVER", ""))
    print("SMTP username      :", username)
    print("From               :", mail_from)
    print("Actual recipient   :", recipient_email)
    print("Subject            :", message["Subject"])
    print("=" * 70)

    with smtplib.SMTP(
        config.MAIL_SERVER,
        int(config.MAIL_PORT),
        timeout=20
    ) as smtp:
        if getattr(config, "MAIL_USE_TLS", True):
            smtp.starttls()
        smtp.login(username, password)
        smtp.send_message(message)

    print("RESET EMAIL ACCEPTED BY SMTP FOR:", recipient_email)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()

        if not email:
            flash("Please enter your registered email address.", "warning")
            return redirect(url_for("forgot_password"))

        cur = None
        try:
            cur = mysql.connection.cursor()
            cur.execute(
                """
                SELECT id, fullname, email
                FROM users
                WHERE LOWER(email) = %s
                LIMIT 1
                """,
                (email,)
            )
            user = cur.fetchone()

            # Keep the public response generic to avoid email enumeration.
            if not user:
                flash(
                    "If an account exists for this email, a password-reset OTP will be sent.",
                    "info"
                )
                return redirect(url_for("forgot_password"))

            user_id, fullname, registered_email = user

            # Prevent accidental duplicate sends when the user repeatedly clicks Send OTP.
            resend_key = f"reset_last_sent_{user_id}"
            last_sent = session.get(resend_key)
            if last_sent:
                try:
                    elapsed = datetime.utcnow().timestamp() - float(last_sent)
                    if elapsed < RESET_RESEND_SECONDS:
                        remaining = max(1, int(RESET_RESEND_SECONDS - elapsed))
                        flash(
                            f"Please wait {remaining} seconds before requesting another OTP.",
                            "warning"
                        )
                        return redirect(url_for("forgot_password"))
                except (TypeError, ValueError):
                    session.pop(resend_key, None)

            otp = generate_reset_otp()
            otp_hash = hash_reset_otp(otp)
            expires_at = datetime.now() + timedelta(minutes=RESET_OTP_EXPIRY_MINUTES)

            # Invalidate all previous unused OTPs for this account.
            cur.execute(
                """
                UPDATE password_reset_otps
                SET used = 1
                WHERE user_id = %s AND used = 0
                """,
                (user_id,)
            )

            cur.execute(
                """
                INSERT INTO password_reset_otps
                    (user_id, otp_hash, expires_at, attempts, used)
                VALUES (%s, %s, %s, 0, 0)
                """,
                (user_id, otp_hash, expires_at)
            )
            mysql.connection.commit()

            print("=" * 70)
            print("PASSWORD RESET DEBUG")
            print("Entered email      :", email)
            print("Database email     :", registered_email)
            print("User ID             :", user_id)
            print("Recipient email     :", registered_email)
            print("OTP generated      :", otp)
            print("OTP expires        :", expires_at)
            print("=" * 70)

            try:
                send_password_reset_email(registered_email, fullname, otp)
            except Exception as mail_error:
                print("PASSWORD RESET EMAIL ERROR:", repr(mail_error))
                try:
                    cur.execute(
                        """
                        DELETE FROM password_reset_otps
                        WHERE user_id = %s AND used = 0
                        """,
                        (user_id,)
                    )
                    mysql.connection.commit()
                except Exception as cleanup_error:
                    print("OTP CLEANUP ERROR:", repr(cleanup_error))

                flash(
                    "We could not send the OTP email. Please verify the email configuration and try again.",
                    "danger"
                )
                return redirect(url_for("forgot_password"))

            session["password_reset_user_id"] = int(user_id)
            session["password_reset_email"] = registered_email
            session["password_reset_verified"] = False
            session[resend_key] = datetime.utcnow().timestamp()

            flash(
                "If an account exists for this email, a password-reset OTP has been sent.",
                "success"
            )
            return redirect(url_for("verify_reset_otp"))

        except Exception as e:
            try:
                mysql.connection.rollback()
            except Exception:
                pass
            print("FORGOT PASSWORD ERROR:", repr(e))
            flash("Unable to process the password-reset request. Please try again.", "danger")
            return redirect(url_for("forgot_password"))
        finally:
            if cur:
                try:
                    cur.close()
                except Exception:
                    pass

    return render_template("forgot_password.html")


@app.route("/verify-reset-otp", methods=["GET", "POST"])
def verify_reset_otp():
    user_id = session.get("password_reset_user_id")
    email = session.get("password_reset_email")

    if not user_id or not email:
        flash("Please request a new password-reset OTP first.", "warning")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        otp = request.form.get("otp", "").strip()

        if not otp.isdigit() or len(otp) != 6:
            flash("Please enter the 6-digit OTP.", "warning")
            return redirect(url_for("verify_reset_otp"))

        cur = None
        try:
            cur = mysql.connection.cursor()
            cur.execute(
                """
                SELECT id, otp_hash, expires_at, attempts, used
                FROM password_reset_otps
                WHERE user_id = %s AND used = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (user_id,)
            )
            row = cur.fetchone()

            if not row:
                flash("This OTP is no longer valid. Please request a new OTP.", "danger")
                return redirect(url_for("forgot_password"))

            reset_id, stored_hash, expires_at, attempts, used = row

            if used:
                flash("This OTP has already been used. Please request a new OTP.", "danger")
                return redirect(url_for("forgot_password"))

            # MySQL returns a datetime object for DATETIME.
            if datetime.now() > expires_at:
                cur.execute(
                    "UPDATE password_reset_otps SET used = 1 WHERE id = %s",
                    (reset_id,)
                )
                mysql.connection.commit()
                flash("The OTP has expired. Please request a new OTP.", "danger")
                return redirect(url_for("forgot_password"))

            if int(attempts or 0) >= RESET_MAX_ATTEMPTS:
                cur.execute(
                    "UPDATE password_reset_otps SET used = 1 WHERE id = %s",
                    (reset_id,)
                )
                mysql.connection.commit()
                flash("Too many incorrect attempts. Please request a new OTP.", "danger")
                return redirect(url_for("forgot_password"))

            if hash_reset_otp(otp) != stored_hash:
                new_attempts = int(attempts or 0) + 1
                cur.execute(
                    "UPDATE password_reset_otps SET attempts = %s WHERE id = %s",
                    (new_attempts, reset_id)
                )
                mysql.connection.commit()
                remaining = max(0, RESET_MAX_ATTEMPTS - new_attempts)
                if remaining:
                    flash(f"Incorrect OTP. {remaining} attempt(s) remaining.", "danger")
                else:
                    cur.execute(
                        "UPDATE password_reset_otps SET used = 1 WHERE id = %s",
                        (reset_id,)
                    )
                    mysql.connection.commit()
                    flash("Too many incorrect attempts. Please request a new OTP.", "danger")
                    return redirect(url_for("forgot_password"))
                return redirect(url_for("verify_reset_otp"))

            # OTP is valid. It is not marked used until the password is successfully changed.
            session["password_reset_verified"] = True
            session["password_reset_otp_id"] = int(reset_id)

            flash("OTP verified successfully. Create your new password.", "success")
            return redirect(url_for("reset_password"))

        except Exception as e:
            try:
                mysql.connection.rollback()
            except Exception:
                pass
            print("VERIFY RESET OTP ERROR:", repr(e))
            flash("Unable to verify the OTP. Please try again.", "danger")
        finally:
            if cur:
                try:
                    cur.close()
                except Exception:
                    pass

    return render_template("verify_reset_otp.html", email=email)


@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    user_id = session.get("password_reset_user_id")
    verified = session.get("password_reset_verified") is True
    reset_id = session.get("password_reset_otp_id")

    if not user_id or not verified or not reset_id:
        flash("Please verify the OTP before creating a new password.", "warning")
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if len(password) < 8:
            flash("Password must contain at least 8 characters.", "warning")
            return redirect(url_for("reset_password"))

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("reset_password"))

        cur = None
        try:
            cur = mysql.connection.cursor()
            cur.execute(
                """
                SELECT id, used, expires_at
                FROM password_reset_otps
                WHERE id = %s AND user_id = %s
                LIMIT 1
                """,
                (reset_id, user_id)
            )
            otp_row = cur.fetchone()

            if not otp_row:
                flash("The password-reset session is invalid. Please request a new OTP.", "danger")
                return redirect(url_for("forgot_password"))

            otp_db_id, used, expires_at = otp_row
            if used or datetime.now() > expires_at:
                flash("The OTP session has expired. Please request a new OTP.", "danger")
                return redirect(url_for("forgot_password"))

            hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

            cur.execute(
                """
                UPDATE users
                SET password = %s
                WHERE id = %s
                """,
                (hashed_password, user_id)
            )

            if cur.rowcount != 1:
                mysql.connection.rollback()
                flash("Unable to update the password. Please try again.", "danger")
                return redirect(url_for("forgot_password"))

            cur.execute(
                "UPDATE password_reset_otps SET used = 1 WHERE id = %s",
                (otp_db_id,)
            )
            mysql.connection.commit()

            # Clear every password-reset session value.
            for key in (
                "password_reset_user_id",
                "password_reset_email",
                "password_reset_verified",
                "password_reset_otp_id",
            ):
                session.pop(key, None)

            flash("Password reset successfully. Please login with your new password.", "success")
            return redirect(url_for("login"))

        except Exception as e:
            try:
                mysql.connection.rollback()
            except Exception:
                pass
            print("RESET PASSWORD ERROR:", repr(e))
            flash("Unable to reset the password. Please try again.", "danger")
        finally:
            if cur:
                try:
                    cur.close()
                except Exception:
                    pass

    return render_template("reset_password.html")


# ============================================================
# ============================================================
# AI HEALTH ASSISTANT API
# ============================================================


# ============================================================

@app.route("/api/health-chat", methods=["POST"])
def health_chat():

    # ========================================================
    # LOGIN CHECK
    # ========================================================

    if not login_required():

        return jsonify({
            "success": False,
            "error": (
                "Please log in to use the "
                "AI Health Assistant."
            )
        }), 401

    # ========================================================
    # READ REQUEST
    # ========================================================

    data = request.get_json(silent=True) or {}

    message = str(
        data.get("message", "")
    ).strip()

    if not message:

        return jsonify({
            "success": False,
            "error": "Please enter a question."
        }), 400

    # ========================================================
    # INITIAL AI CONTEXT
    # ========================================================

    context = {
        "total_predictions": 0,
        "medical_reports": 0,
        "latest_prediction": None,
        "latest_diabetes_prediction": None,
        "latest_early_diabetes_prediction": None,
        "prediction_history": []
    }

    cur = None

    try:

        cur = mysql.connection.cursor()

        # ====================================================
        # RESOLVE LOGGED-IN USER
        # ====================================================

        session_user_id = session.get("user_id")
        session_email = session.get("email")

        resolved_user_id = session_user_id

        print(
            "AI CHAT SESSION:",
            {
                "session_user_id": session_user_id,
                "session_email": session_email
            }
        )

        if session_email:

            cur.execute(
                """
                SELECT id
                FROM users
                WHERE email = %s
                LIMIT 1
                """,
                (session_email,)
            )

            user_row = cur.fetchone()

            if user_row:

                resolved_user_id = user_row[0]

                session["user_id"] = (
                    resolved_user_id
                )

                print(
                    "AI CHAT RESOLVED USER ID:",
                    resolved_user_id
                )

        if resolved_user_id is None:

            raise RuntimeError(
                "No authenticated user ID could be resolved."
            )

        # ====================================================
        # TOTAL PREDICTIONS
        # ====================================================

        cur.execute(
            """
            SELECT COUNT(*)
            FROM predictions
            WHERE user_id = %s
            """,
            (resolved_user_id,)
        )

        count_row = cur.fetchone()

        context["total_predictions"] = int(
            count_row[0]
            if count_row
            else 0
        )

        # ====================================================
        # LATEST PREDICTION
        # ====================================================

        cur.execute(
            """
            SELECT
                id,
                disease_name,
                prediction,
                confidence,
                high_risk_probability,
                prediction_date,
                input_data
            FROM predictions
            WHERE user_id = %s
            ORDER BY
                prediction_date DESC,
                id DESC
            LIMIT 1
            """,
            (resolved_user_id,)
        )

        row = cur.fetchone()

        print(
            "AI CHAT LATEST PREDICTION:",
            row
        )

        if row:

            stored_input_data = None

            if row[6]:

                try:

                    if isinstance(row[6], str):

                        stored_input_data = (
                            json.loads(row[6])
                        )

                    elif isinstance(row[6], dict):

                        stored_input_data = (
                            row[6]
                        )

                except (
                    TypeError,
                    ValueError,
                    json.JSONDecodeError
                ) as json_error:

                    print(
                        "AI CHAT INPUT JSON ERROR:",
                        json_error
                    )

            context["latest_prediction"] = {

                "prediction_id":
                    row[0],

                "disease_name":
                    row[1],

                "prediction":
                    row[2],

                "confidence":
                    row[3],

                "high_risk_probability":
                    row[4],

                "prediction_date":
                    str(row[5]),

                "input_data":
                    stored_input_data
            }

        # ====================================================
        # LATEST HEART PREDICTION FOR HEART CHAT
        # ====================================================

        cur.execute(
            """SELECT id, disease_name, prediction, confidence, high_risk_probability, prediction_date, input_data
               FROM predictions
               WHERE user_id = %s AND LOWER(TRIM(disease_name)) IN ('heart disease','heart','heart_disease')
               ORDER BY prediction_date DESC, id DESC LIMIT 1""",
            (resolved_user_id,)
        )
        heart_row = cur.fetchone()
        if heart_row:
            try:
                heart_input_data = json.loads(heart_row[6]) if heart_row[6] else {}
            except Exception:
                heart_input_data = {}
            context["latest_heart_prediction"] = {
                "prediction_id": heart_row[0], "disease_name": heart_row[1],
                "prediction": heart_row[2], "confidence": heart_row[3],
                "high_risk_probability": heart_row[4], "prediction_date": str(heart_row[5]),
                "input_data": heart_input_data
            }
        else:
            context["latest_heart_prediction"] = None

        # ====================================================
        # LATEST DIABETES PREDICTION FOR DIABETES CHAT
        # ====================================================

        cur.execute(
            """SELECT id, disease_name, prediction, confidence, high_risk_probability, prediction_date, input_data
               FROM predictions
               WHERE user_id = %s AND LOWER(disease_name) = 'diabetes'
               ORDER BY prediction_date DESC, id DESC LIMIT 1""",
            (resolved_user_id,)
        )
        diabetes_row = cur.fetchone()
        if diabetes_row:
            try:
                diabetes_input_data = json.loads(diabetes_row[6]) if diabetes_row[6] else {}
            except Exception:
                diabetes_input_data = {}
            context["latest_diabetes_prediction"] = {
                "prediction_id": diabetes_row[0], "disease_name": diabetes_row[1],
                "prediction": diabetes_row[2], "confidence": diabetes_row[3],
                "high_risk_probability": diabetes_row[4], "prediction_date": str(diabetes_row[5]),
                "input_data": diabetes_input_data
            }

        # ====================================================
        # LATEST EARLY DIABETES PREDICTION FOR CHAT
        # ====================================================

        cur.execute(
            """SELECT id, disease_name, prediction, confidence, high_risk_probability, prediction_date, input_data
               FROM predictions
               WHERE user_id = %s AND LOWER(TRIM(disease_name)) IN ('early diabetes','early_diabetes','early diabetes analysis')
               ORDER BY prediction_date DESC, id DESC LIMIT 1""",
            (resolved_user_id,)
        )
        early_diabetes_row = cur.fetchone()
        if early_diabetes_row:
            try:
                early_diabetes_input_data = json.loads(early_diabetes_row[6]) if early_diabetes_row[6] else {}
            except Exception:
                early_diabetes_input_data = {}
            context["latest_early_diabetes_prediction"] = {
                "prediction_id": early_diabetes_row[0],
                "disease_name": early_diabetes_row[1],
                "prediction": early_diabetes_row[2],
                "confidence": early_diabetes_row[3],
                "high_risk_probability": early_diabetes_row[4],
                "prediction_date": str(early_diabetes_row[5]),
                "input_data": early_diabetes_input_data
            }
        else:
            context["latest_early_diabetes_prediction"] = None

        # ====================================================
        # PREDICTION HISTORY
        # ====================================================

        cur.execute(
            """
            SELECT
                id,
                disease_name,
                prediction,
                confidence,
                high_risk_probability,
                prediction_date
            FROM predictions
            WHERE user_id = %s
            ORDER BY
                prediction_date DESC,
                id DESC
            LIMIT 10
            """,
            (resolved_user_id,)
        )

        history_rows = cur.fetchall()

        context["prediction_history"] = []

        for history_row in history_rows:

            context["prediction_history"].append({

                "prediction_id":
                    history_row[0],

                "disease_name":
                    history_row[1],

                "prediction":
                    history_row[2],

                "confidence":
                    history_row[3],

                "high_risk_probability":
                    history_row[4],

                "prediction_date":
                    str(history_row[5])
            })

        # ====================================================
        # MEDICAL REPORT COUNT
        # ====================================================

        try:

            cur.execute(
                """
                SELECT COUNT(*)
                FROM medical_reports
                WHERE user_id = %s
                """,
                (resolved_user_id,)
            )

            report_row = cur.fetchone()

            context["medical_reports"] = int(
                report_row[0]
                if report_row
                else 0
            )

        except Exception as report_error:

            mysql.connection.rollback()

            print(
                "AI CHAT MEDICAL REPORT COUNT WARNING:",
                report_error
            )

            context["medical_reports"] = 0

    except Exception as e:

        print(
            "AI CONTEXT ERROR:",
            e
        )

        try:
            mysql.connection.rollback()
        except Exception:
            pass

    finally:

        if cur:
            cur.close()

    # ========================================================
    # DEBUG
    # ========================================================

    print("=" * 70)
    print("AI HEALTH ASSISTANT CONTEXT")
    print(
        "Total predictions:",
        context["total_predictions"]
    )
    print(
        "Latest prediction:",
        context["latest_prediction"]
    )
    print(
        "Prediction history count:",
        len(context.get("prediction_history", []))
    )
    print(
        "Prediction history:",
        context.get("prediction_history", [])
    )
    print(
        "Medical reports:",
        context["medical_reports"]
    )
    print("=" * 70)

    # ========================================================
    # ASK AI HEALTH ASSISTANT
    # ========================================================

    try:

        answer = ask_health_assistant(
            message,
            context
        )

        latest = (
            context.get("latest_prediction")
            or {}
        )

        return jsonify({

            "success":
                True,

            "answer":
                answer,

            "context_used":
                bool(latest),

            "prediction_id":
                latest.get("prediction_id"),

            "total_predictions":
                context.get(
                    "total_predictions",
                    0
                )
        })

    except requests.exceptions.ConnectionError:

        return jsonify({
            "success": False,
            "error": (
                "Ollama is not running. "
                "Start Ollama and try again."
            )
        }), 503

    except requests.exceptions.Timeout:

        return jsonify({
            "success": False,
            "error": (
                "The AI assistant took too "
                "long to respond. Please try again."
            )
        }), 504

    except Exception as e:

        print(
            "AI HEALTH ASSISTANT ERROR:",
            e
        )

        return jsonify({
            "success": False,
            "error": (
                "The AI Health Assistant is "
                "temporarily unavailable."
            )
        }), 500


# ============================================================
# HEALTH RISK TREND API
# ============================================================

@app.route("/api/risk-trend")
def risk_trend_api():

    if not login_required():
        return jsonify({"success": False, "error": "Login required."}), 401

    user_id = session["user_id"]
    cur = mysql.connection.cursor()

    try:
        cur.execute(
            """
            SELECT
                id,
                disease_name,
                prediction,
                confidence,
                high_risk_probability,
                prediction_date
            FROM predictions
            WHERE user_id = %s
            ORDER BY prediction_date ASC, id ASC
            LIMIT 50
            """,
            (user_id,)
        )

        rows = cur.fetchall()

        trend = []

        for row in rows:
            prediction_id, disease_name, prediction, confidence, high_risk_probability, prediction_date = row
            confidence = float(confidence or 0)
            prediction_text = str(prediction or "").strip()
            risk_score = row_high_risk_probability(prediction_text, confidence, high_risk_probability)

            trend.append({
                "id": prediction_id,
                "disease": disease_name or "Unknown",
                "prediction": prediction_text,
                "confidence": round(confidence, 2),
                "predicted_class_probability": round(confidence, 2),
                "high_risk_probability": round(risk_score, 2),
                "risk_score": round(risk_score, 2),
                "date": prediction_date.strftime("%d %b %Y") if prediction_date else "",
                "timestamp": prediction_date.isoformat() if prediction_date else ""
            })

        # Keep the response focused on the most recent 12 assessments.
        trend = trend[-12:]

        latest = trend[-1] if trend else None
        previous = trend[-2] if len(trend) >= 2 else None

        change = None
        direction = "insufficient_data"

        if latest and previous:
            change = round(latest["risk_score"] - previous["risk_score"], 2)
            if change >= 5:
                direction = "increasing"
            elif change <= -5:
                direction = "decreasing"
            else:
                direction = "stable"

        return jsonify({
            "success": True,
            "trend": trend,
            "latest": latest,
            "previous": previous,
            "change": change,
            "direction": direction,
            "total_available": len(rows)
        })

    except Exception as e:
        mysql.connection.rollback()
        print("RISK TREND API ERROR:", e)
        return jsonify({
            "success": False,
            "error": f"Unable to load risk trend data: {str(e)}"
        }), 500

    finally:
        cur.close()


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    if not login_required():
        return redirect(url_for("login"))

    user_id = session["user_id"]

    cur = mysql.connection.cursor()

    # --------------------------------------------------------
    # TOTAL PREDICTIONS
    # --------------------------------------------------------

    cur.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id = %s
        """,
        (user_id,)
    )

    total_predictions = cur.fetchone()[0]


    # --------------------------------------------------------
    # HEART ASSESSMENTS
    # --------------------------------------------------------

    cur.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id = %s
        AND disease_name = 'Heart Disease'
        """,
        (user_id,)
    )

    heart_assessments = cur.fetchone()[0]


    cur.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id = %s
        AND disease_name = 'Diabetes'
        """,
        (user_id,)
    )

    diabetes_assessments = cur.fetchone()[0]


    # --------------------------------------------------------
    # HIGH RISK PREDICTIONS
    # --------------------------------------------------------

    cur.execute(
        """
        SELECT COUNT(*)
        FROM predictions
        WHERE user_id = %s
        AND prediction LIKE '%%High%%'
        """,
        (user_id,)
    )

    high_risk_predictions = cur.fetchone()[0]


    # --------------------------------------------------------
    # MEDICAL REPORTS
    # --------------------------------------------------------

    try:

        cur.execute(
            """
            SELECT COUNT(*)
            FROM medical_reports
            WHERE user_id = %s
            """,
            (user_id,)
        )

        medical_reports = cur.fetchone()[0]

    except Exception:

        mysql.connection.rollback()

        medical_reports = 0


    # --------------------------------------------------------
    # LATEST PREDICTION
    # --------------------------------------------------------

    latest_prediction = None

    try:

        cur.execute(
            """
            SELECT
                disease_name,
                prediction,
                confidence,
                prediction_date
            FROM predictions
            WHERE user_id = %s
            ORDER BY prediction_date DESC
            LIMIT 1
            """,
            (user_id,)
        )

        latest_prediction = cur.fetchone()

    except Exception:

        mysql.connection.rollback()


    # --------------------------------------------------------
    # PHASE 16 PROFESSIONAL INTELLIGENCE SUMMARY
    # --------------------------------------------------------
    phase16_cards = []
    for disease_name, title, icon in (("Heart Disease","Heart Disease","❤️"),("Diabetes","Diabetes","🩸"),("Early Diabetes","Early Diabetes","🔎")):
        cur.execute("""SELECT id,prediction,confidence,high_risk_probability,prediction_date
                       FROM predictions WHERE user_id=%s AND disease_name=%s
                       ORDER BY prediction_date DESC,id DESC LIMIT 2""", (user_id,disease_name))
        rs=cur.fetchall()
        latest=None; delta=None; direction="No assessment yet"
        if rs:
            latest={"id":rs[0][0],"prediction":rs[0][1],"probability":round(float(row_high_risk_probability(rs[0][1],rs[0][2],rs[0][3])),2),
                    "date":rs[0][4].strftime("%d %b %Y") if hasattr(rs[0][4],"strftime") else str(rs[0][4])}
            direction="First saved assessment"
            if len(rs)>1:
                prev=float(row_high_risk_probability(rs[1][1],rs[1][2],rs[1][3]))
                delta=round(latest["probability"]-prev,2)
                direction="Increasing" if delta>=5 else ("Decreasing" if delta<=-5 else "Relatively stable")
        phase16_cards.append({"disease":disease_name,"title":title,"icon":icon,"latest":latest,"delta":delta,"direction":direction})

    # --------------------------------------------------------
    # CLOSE CURSOR
    # --------------------------------------------------------

    cur.close()


    # --------------------------------------------------------
    # RENDER DASHBOARD
    # --------------------------------------------------------

    return render_template(
        "dashboard.html",

        total_predictions=total_predictions,

        heart_assessments=heart_assessments,

        diabetes_assessments=diabetes_assessments,

        medical_reports=medical_reports,

        high_risk_predictions=high_risk_predictions,

        latest_prediction=latest_prediction,
        phase16_cards=phase16_cards
    )
# ============================================================
# HEALTHCARE
# ============================================================

@app.route("/healthcare")
def healthcare():

    if "user_id" not in session:

        return redirect(
            url_for("login")
        )


    return render_template(
        "healthcare.html"
    )


# ============================================================
# EXTRACTION TRACEABILITY
# ============================================================

def _trace_value_equal(a, b):
    """Compare extracted and submitted values without false differences from numeric types."""
    if a is None or b is None:
        return a is None and b is None
    try:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) < 1e-9
    except Exception:
        pass
    return str(a).strip().lower() == str(b).strip().lower()


def build_extraction_trace(user_data, report_id):
    """Return field-level provenance for the values actually sent to the model."""
    fields = {}
    extracted = {}
    if report_id:
        try:
            cur = mysql.connection.cursor()
            cur.execute(
                "SELECT extracted_data FROM medical_reports WHERE id = %s AND user_id = %s",
                (int(report_id), session["user_id"])
            )
            row = cur.fetchone()
            cur.close()
            if row and row[0]:
                extracted = json.loads(row[0])
        except Exception as exc:
            print("EXTRACTION TRACE ERROR:", exc)
            try:
                mysql.connection.rollback()
            except Exception:
                pass

    if not extracted:
        for field in EXPECTED_FIELDS:
            fields[field] = {"status": "manual", "label": "Manual entry"}
        return {"summary": "Manual assessment entry", "fields": fields, "report_id": None}

    changed = 0
    auto_verified = 0
    manual = 0
    for field in EXPECTED_FIELDS:
        original = extracted.get(field)
        final = user_data.get(field)
        if original is None:
            status = "manual"
            label = "Manually entered (not found in report)"
            manual += 1
        elif _trace_value_equal(original, final):
            status = "verified"
            label = "Automatically extracted · user verified"
            auto_verified += 1
        else:
            status = "edited"
            label = "Automatically extracted · user modified"
            changed += 1
        fields[field] = {"status": status, "label": label, "extracted": original, "final": final}

    if changed:
        summary = f"Medical report extraction + user verification ({changed} value{'s' if changed != 1 else ''} modified)"
    elif manual:
        summary = f"Medical report extraction + user verification ({manual} value{'s' if manual != 1 else ''} entered manually)"
    else:
        summary = "Medical report extraction + user verification (all extracted values unchanged)"
    return {"summary": summary, "fields": fields, "report_id": int(report_id)}



# ============================================================
# PHASE 16 - PREDICTION AUDIT / MODEL VERSIONING
# ============================================================

PHASE16_MODEL_REGISTRY = {
    "heart": {
        "model_name": "Adaptive Ensemble (Heart Disease)",
        "model_version": "HEART-AE-2026.16",
        "model_file": "models/heart/ensemble_model.pkl",
        "dataset_reference": "dataset/heart.csv (918 records / 11 predictors)",
        "threshold": 0.50,
    },
    "diabetes": {
        "model_name": "Adaptive Weighted Ensemble (Diabetes)",
        "model_version": "DIABETES-AE-2026.16",
        "model_file": "runtime diabetes ensemble",
        "dataset_reference": "Pima-style diabetes dataset / 8 predictors",
        "threshold": 0.50,
    },
    "early_diabetes": {
        "model_name": "Adaptive Weighted Ensemble (Early Diabetes)",
        "model_version": "EARLY-DIABETES-AE-2026.16",
        "model_file": "runtime early-diabetes ensemble",
        "dataset_reference": "Early Diabetes symptom dataset / 16 predictors",
        "threshold": 0.50,
    },
}


def _phase16_registry_key(disease_name):
    normalized = str(disease_name or "").strip().lower().replace("_", " ")
    if normalized in ("heart", "heart disease"):
        return "heart"
    if normalized == "diabetes":
        return "diabetes"
    if normalized == "early diabetes":
        return "early_diabetes"
    return None


def _phase16_write_audit(prediction_id, user_id, disease_name, prediction, probability, input_data):
    """Persist a reproducible model/version snapshot. Prediction saving must never fail because of audit logging."""
    key = _phase16_registry_key(disease_name)
    meta = PHASE16_MODEL_REGISTRY.get(key, {})
    cur = None
    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """INSERT INTO prediction_audit
               (prediction_id,user_id,disease_name,model_name,model_version,model_file,
                dataset_reference,decision_threshold,high_risk_probability,predicted_label,input_snapshot)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON DUPLICATE KEY UPDATE
                 model_name=VALUES(model_name), model_version=VALUES(model_version),
                 model_file=VALUES(model_file), dataset_reference=VALUES(dataset_reference),
                 decision_threshold=VALUES(decision_threshold), high_risk_probability=VALUES(high_risk_probability),
                 predicted_label=VALUES(predicted_label), input_snapshot=VALUES(input_snapshot)""",
            (prediction_id, user_id, disease_name, meta.get("model_name"), meta.get("model_version"),
             meta.get("model_file"), meta.get("dataset_reference"), meta.get("threshold", 0.50),
             float(probability), prediction, json.dumps(input_data, default=str))
        )
        mysql.connection.commit()
    except Exception as audit_error:
        try:
            mysql.connection.rollback()
        except Exception:
            pass
        print("PREDICTION AUDIT WARNING:", audit_error)
    finally:
        if cur:
            try: cur.close()
            except Exception: pass


def _phase16_get_audit(prediction_id, user_id, disease_name=None, prediction=None, probability=None, input_data=None):
    cur = None
    try:
        cur = mysql.connection.cursor()
        cur.execute("""SELECT model_name,model_version,model_file,dataset_reference,decision_threshold,
                              high_risk_probability,predicted_label,audit_created_at
                       FROM prediction_audit WHERE prediction_id=%s AND user_id=%s LIMIT 1""",
                    (prediction_id, user_id))
        row = cur.fetchone()
        if row:
            return {"model_name":row[0],"model_version":row[1],"model_file":row[2],"dataset_reference":row[3],
                    "decision_threshold":float(row[4] or .5),"probability":float(row[5] or 0),
                    "predicted_label":row[6],"created_at":row[7].strftime("%d %b %Y, %I:%M %p") if hasattr(row[7],"strftime") else str(row[7])}
    except Exception as audit_error:
        print("AUDIT READ WARNING:", audit_error)
    finally:
        if cur:
            try: cur.close()
            except Exception: pass
    # Backfill older saved assessments lazily so previous data remains usable.
    if disease_name is not None and probability is not None:
        _phase16_write_audit(prediction_id, user_id, disease_name, prediction or "", probability, input_data or {})
        return _phase16_get_audit(prediction_id, user_id)
    return None


# ============================================================
# PHASE 16A - ADVANCED PREVENTIVE HEALTH INTELLIGENCE
# Longitudinal trends + non-persistent what-if simulation +
# ensemble disagreement / uncertainty view.
# ============================================================

PHASE16_FIELD_CONFIG = {
    "heart": [
        ("age", "Age", "number", "years"), ("sex", "Sex", "select", ""),
        ("chest_pain_type", "Chest Pain Type", "select", ""),
        ("resting_bp", "Resting Blood Pressure", "number", "mmHg"),
        ("cholesterol", "Cholesterol", "number", "mg/dL"),
        ("fasting_bs", "Fasting Blood Sugar", "select", ""),
        ("resting_ecg", "Resting ECG", "select", ""),
        ("max_hr", "Maximum Heart Rate", "number", "bpm"),
        ("exercise_angina", "Exercise Angina", "select", ""),
        ("oldpeak", "Oldpeak", "number", ""), ("st_slope", "ST Slope", "select", ""),
    ],
    "diabetes": [
        ("pregnancies", "Pregnancies", "number", ""), ("glucose", "Glucose", "number", "mg/dL"),
        ("blood_pressure", "Blood Pressure", "number", "mmHg"),
        ("skin_thickness", "Skin Thickness", "number", "mm"), ("insulin", "Insulin", "number", "µU/mL"),
        ("bmi", "BMI", "number", "kg/m²"),
        ("diabetes_pedigree_function", "Diabetes Pedigree Function", "number", ""),
        ("age", "Age", "number", "years"),
    ],
    "early_diabetes": [
        ("age", "Age", "number", "years"), ("gender", "Gender", "select", ""),
        ("polyuria", "Polyuria", "select", ""), ("polydipsia", "Polydipsia", "select", ""),
        ("sudden_weight_loss", "Sudden Weight Loss", "select", ""), ("weakness", "Weakness", "select", ""),
        ("polyphagia", "Polyphagia", "select", ""), ("genital_thrush", "Genital Thrush", "select", ""),
        ("visual_blurring", "Visual Blurring", "select", ""), ("itching", "Itching", "select", ""),
        ("irritability", "Irritability", "select", ""), ("delayed_healing", "Delayed Healing", "select", ""),
        ("partial_paresis", "Partial Paresis", "select", ""), ("muscle_stiffness", "Muscle Stiffness", "select", ""),
        ("alopecia", "Alopecia", "select", ""), ("obesity", "Obesity", "select", ""),
    ],
}

PHASE16_SELECT_OPTIONS = {
    "sex": ["M", "F"], "chest_pain_type": ["TA", "ATA", "NAP", "ASY"],
    "fasting_bs": ["0", "1"], "resting_ecg": ["Normal", "ST", "LVH"],
    "exercise_angina": ["N", "Y"], "st_slope": ["Up", "Flat", "Down"],
    "gender": ["Male", "Female"],
}
for _phase16_symptom in EARLY_FIELDS[2:]:
    PHASE16_SELECT_OPTIONS[_phase16_symptom] = ["Yes", "No"]


def _phase16_disease_key(name):
    normalized = str(name or "").strip().lower().replace("_", " ")
    if normalized in ("heart", "heart disease"):
        return "heart"
    if normalized == "diabetes":
        return "diabetes"
    if normalized == "early diabetes":
        return "early_diabetes"
    return None


def _phase16_model_output(disease_key, data):
    """Run a hypothetical input through the existing model without saving it."""
    if disease_key == "heart":
        clean = {
            "age": float(data["age"]), "sex": str(data["sex"]),
            "chest_pain_type": str(data["chest_pain_type"]), "resting_bp": float(data["resting_bp"]),
            "cholesterol": float(data["cholesterol"]), "fasting_bs": int(float(data["fasting_bs"])),
            "resting_ecg": str(data["resting_ecg"]), "max_hr": float(data["max_hr"]),
            "exercise_angina": str(data["exercise_angina"]), "oldpeak": float(data["oldpeak"]),
            "st_slope": str(data["st_slope"]),
        }
        frame = pd.DataFrame([[clean[f] for f in EXPECTED_FIELDS]], columns=MODEL_COLUMNS)
        probability = heart_model.predict_proba(frame)[0]
        details = heart_model.model_details(frame)[0] if hasattr(heart_model, "model_details") else []
        return clean, round(float(probability[1]) * 100, 2), details
    if disease_key == "diabetes":
        clean = {f: float(data[f]) for f in DIABETES_FIELDS}
        output = diabetes_engine.predict(clean)
        return clean, float(output["high_risk_probability"]), output.get("model_details", [])
    if disease_key == "early_diabetes":
        clean = {f: data[f] for f in EARLY_FIELDS}
        clean["age"] = float(clean["age"])
        output = early_diabetes_engine.predict(clean)
        return clean, float(output["high_risk_probability"]), output.get("model_details", [])
    raise ValueError("Unsupported disease module for Phase 16 intelligence.")


def _phase16_disagreement(details):
    probabilities = [float(x.get("high_risk_probability", 0)) for x in (details or [])]
    if not probabilities:
        return {"level": "Unavailable", "spread": None, "message": "Per-model probability information is unavailable."}
    spread = round(max(probabilities) - min(probabilities), 2)
    if spread < 15:
        level = "Low"
    elif spread < 30:
        level = "Moderate"
    else:
        level = "High"
    return {
        "level": level, "spread": spread,
        "message": "Model disagreement is descriptive. A larger spread means the component models respond less consistently to this input and the ensemble result should be interpreted with greater caution."
    }


@app.route("/advanced-intelligence/<int:prediction_id>", methods=["GET", "POST"])
def advanced_intelligence(prediction_id):
    if not login_required():
        return redirect(url_for("login"))

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT id, disease_name, prediction, confidence, high_risk_probability, input_data, prediction_date
        FROM predictions WHERE id=%s AND user_id=%s LIMIT 1
    """, (prediction_id, session["user_id"]))
    row = cur.fetchone()
    if not row:
        cur.close()
        flash("Assessment not found.", "danger")
        return redirect(url_for("history"))

    disease_key = _phase16_disease_key(row[1])
    if not disease_key:
        cur.close()
        flash("Advanced intelligence is not available for this assessment type yet.", "warning")
        return redirect(url_for("history"))

    current_inputs = _load_prediction_input(row[5])
    current_probability = row_high_risk_probability(row[2], row[3], row[4])
    audit = _phase16_get_audit(prediction_id, session["user_id"], row[1], row[2], current_probability, current_inputs)

    # Disease-specific longitudinal history only.
    aliases = ("Heart Disease", "Heart", "heart_disease") if disease_key == "heart" else (("Diabetes",) if disease_key == "diabetes" else ("Early Diabetes",))
    placeholders = ",".join(["%s"] * len(aliases))
    cur.execute(f"""
        SELECT id, prediction, confidence, high_risk_probability, prediction_date
        FROM predictions
        WHERE user_id=%s AND disease_name IN ({placeholders})
        ORDER BY prediction_date ASC, id ASC
    """, (session["user_id"], *aliases))
    history_rows = cur.fetchall()
    cur.close()
    trend = [{
        "id": r[0], "prediction": r[1],
        "probability": round(float(row_high_risk_probability(r[1], r[2], r[3])), 2),
        "date": r[4].strftime("%d %b %Y") if hasattr(r[4], "strftime") else str(r[4])
    } for r in history_rows]

    trend_direction = "Insufficient history"
    trend_delta = None
    if len(trend) >= 2:
        trend_delta = round(trend[-1]["probability"] - trend[-2]["probability"], 2)
        if trend_delta <= -5:
            trend_direction = "Decreasing model-estimated risk"
        elif trend_delta >= 5:
            trend_direction = "Increasing model-estimated risk"
        else:
            trend_direction = "Relatively stable"

    # Current per-model disagreement uses the existing model, never writes to DB.
    _, _, current_details = _phase16_model_output(disease_key, current_inputs)
    disagreement = _phase16_disagreement(current_details)

    simulation = None
    if request.method == "POST":
        try:
            submitted = {}
            for key, label, input_type, unit in PHASE16_FIELD_CONFIG[disease_key]:
                value = request.form.get(key, "").strip()
                if value == "":
                    raise ValueError(f"Missing value for {label}")
                submitted[key] = value
            clean, simulated_probability, simulated_details = _phase16_model_output(disease_key, submitted)
            simulation = {
                "inputs": clean,
                "probability": simulated_probability,
                "delta": round(simulated_probability - float(current_probability), 2),
                "details": simulated_details,
                "disagreement": _phase16_disagreement(simulated_details),
            }
        except Exception as e:
            flash(f"Unable to run the what-if simulation: {e}", "danger")

    disease_title = {"heart": "Heart Disease", "diabetes": "Diabetes", "early_diabetes": "Early Diabetes"}[disease_key]
    return render_template(
        "advanced_intelligence.html", prediction_id=prediction_id, disease_key=disease_key,
        disease_title=disease_title, current_prediction=row[2], current_probability=round(float(current_probability), 2),
        current_inputs=current_inputs, fields=PHASE16_FIELD_CONFIG[disease_key], select_options=PHASE16_SELECT_OPTIONS,
        trend=trend, trend_direction=trend_direction, trend_delta=trend_delta,
        current_details=current_details, disagreement=disagreement, simulation=simulation, audit=audit,
        prediction_date=row[6].strftime("%d %B %Y, %I:%M %p") if hasattr(row[6], "strftime") else str(row[6])
    )



# ============================================================
# PHASE 16B - CROSS-DISEASE HEALTH INTELLIGENCE
# ============================================================

def _phase16b_num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _phase16b_signal_status(values, rule):
    available = [(name, value) for name, value in values if value not in (None, "", "N/A")]
    if not available:
        return "Unavailable", []
    triggered = [(name, value) for name, value in available if rule(name, value)]
    return ("Attention" if triggered else "No project-rule signal"), triggered


@app.route("/cross-disease-intelligence")
def cross_disease_intelligence():
    if not login_required():
        return redirect(url_for("login"))

    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT id, disease_name, prediction, confidence, high_risk_probability,
               input_data, prediction_date
        FROM predictions
        WHERE user_id=%s
        ORDER BY prediction_date ASC, id ASC
    """, (session["user_id"],))
    rows = cur.fetchall()
    cur.close()

    buckets = {"heart": [], "diabetes": [], "early_diabetes": []}
    for r in rows:
        key = _phase16_disease_key(r[1])
        if key not in buckets:
            continue
        buckets[key].append({
            "id": r[0], "disease_name": r[1], "prediction": r[2],
            "probability": round(float(row_high_risk_probability(r[2], r[3], r[4])), 2),
            "inputs": _load_prediction_input(r[5]),
            "date": r[6].strftime("%d %b %Y") if hasattr(r[6], "strftime") else str(r[6])
        })

    titles = {"heart": "Heart Disease", "diabetes": "Diabetes", "early_diabetes": "Early Diabetes"}
    icons = {"heart": "❤️", "diabetes": "🩸", "early_diabetes": "🔎"}
    disease_cards = []
    for key in ("heart", "diabetes", "early_diabetes"):
        items = buckets[key]
        latest = items[-1] if items else None
        delta = None
        direction = "Insufficient history"
        if len(items) >= 2:
            delta = round(items[-1]["probability"] - items[-2]["probability"], 2)
            direction = "Increasing" if delta >= 5 else ("Decreasing" if delta <= -5 else "Relatively stable")
        disease_cards.append({
            "key": key, "title": titles[key], "icon": icons[key], "count": len(items),
            "latest": latest, "delta": delta, "direction": direction,
            "labels": [x["date"] for x in items[-10:]],
            "values": [x["probability"] for x in items[-10:]],
        })

    heart = buckets["heart"][-1]["inputs"] if buckets["heart"] else {}
    diabetes = buckets["diabetes"][-1]["inputs"] if buckets["diabetes"] else {}
    early = buckets["early_diabetes"][-1]["inputs"] if buckets["early_diabetes"] else {}

    shared_signals = []

    age_values = [
        ("Heart Disease", _phase16b_num(heart.get("age"))),
        ("Diabetes", _phase16b_num(diabetes.get("age"))),
        ("Early Diabetes", _phase16b_num(early.get("age"))),
    ]
    status, triggered = _phase16b_signal_status(age_values, lambda n, v: float(v) >= (60 if n == "Heart Disease" else 45))
    shared_signals.append({"name": "Age-related signal", "icon": "🎂", "status": status, "values": age_values, "triggered": triggered,
                           "note": "Uses the project's existing age attention rules; age is non-modifiable and thresholds differ by module."})

    bp_values = [
        ("Heart Disease", _phase16b_num(heart.get("resting_bp"))),
        ("Diabetes", _phase16b_num(diabetes.get("blood_pressure"))),
    ]
    status, triggered = _phase16b_signal_status(bp_values, lambda n, v: float(v) >= 140)
    shared_signals.append({"name": "Blood-pressure input", "icon": "🫀", "status": status, "values": bp_values, "triggered": triggered,
                           "note": "Compares the latest BP-related inputs used by the Heart and Diabetes modules; it is not a diagnosis of hypertension."})

    weight_values = [
        ("Diabetes BMI", _phase16b_num(diabetes.get("bmi"))),
        ("Early Diabetes obesity", early.get("obesity")),
    ]
    def weight_rule(name, value):
        if name == "Diabetes BMI":
            return float(value) >= 30
        return str(value).strip().lower() in ("yes", "1", "true")
    status, triggered = _phase16b_signal_status(weight_values, weight_rule)
    shared_signals.append({"name": "Weight / metabolic signal", "icon": "⚖️", "status": status, "values": weight_values, "triggered": triggered,
                           "note": "Brings together BMI from the Diabetes model and the obesity symptom input from Early Diabetes without treating them as equivalent measurements."})

    coverage = sum(1 for x in disease_cards if x["latest"])
    latest_probs = [x["latest"]["probability"] for x in disease_cards if x["latest"]]
    highest = max((x for x in disease_cards if x["latest"]), key=lambda x: x["latest"]["probability"], default=None)

    # A research-safe descriptive index: simple mean of available model probabilities.
    # It is deliberately NOT labelled a clinical risk score.
    descriptive_index = round(sum(latest_probs) / len(latest_probs), 2) if latest_probs else None

    return render_template(
        "cross_disease_intelligence.html",
        disease_cards=disease_cards,
        shared_signals=shared_signals,
        coverage=coverage,
        descriptive_index=descriptive_index,
        highest=highest,
        total_assessments=sum(len(v) for v in buckets.values()),
    )


@app.route("/prediction-audit/<int:prediction_id>")
def prediction_audit(prediction_id):
    if not login_required():
        return redirect(url_for("login"))
    cur = mysql.connection.cursor()
    cur.execute("""SELECT id,disease_name,prediction,confidence,high_risk_probability,input_data,prediction_date
                   FROM predictions WHERE id=%s AND user_id=%s LIMIT 1""", (prediction_id, session["user_id"]))
    row = cur.fetchone(); cur.close()
    if not row:
        flash("Assessment not found.", "danger")
        return redirect(url_for("history"))
    probability = row_high_risk_probability(row[2], row[3], row[4])
    inputs = _load_prediction_input(row[5])
    audit = _phase16_get_audit(row[0], session["user_id"], row[1], row[2], probability, inputs)
    return render_template("prediction_audit.html", assessment={
        "id":row[0],"disease":row[1],"prediction":row[2],"confidence":float(row[3] or 0),
        "probability":float(probability),"inputs":inputs,
        "date":row[6].strftime("%d %B %Y, %I:%M %p") if hasattr(row[6],"strftime") else str(row[6])
    }, audit=audit)


# ============================================================
# PHASE 9 - SMART HEALTH ALERT ENGINE
# ============================================================

def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_prediction_input(raw):
    if not raw:
        return {}
    try:
        if isinstance(raw, dict):
            return raw
        return json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def generate_smart_health_alerts(user_id, prediction_id, user_data, prediction,
                                 probability, report_id=None):
    """Create explainable, non-diagnostic alerts by comparing the current
    assessment with the user's immediately previous saved assessment.

    Alerts describe changes in model outputs or submitted assessment values.
    They are not clinical diagnoses and do not infer disease from a single
    parameter change.
    """
    alerts = []
    current_probability = round(float(probability[1]) * 100, 2)
    previous = None
    cur = None

    try:
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT id, prediction, confidence, high_risk_probability, prediction_date, input_data
            FROM predictions
            WHERE user_id = %s AND id <> %s
            ORDER BY prediction_date DESC, id DESC
            LIMIT 1
        """, (user_id, prediction_id))
        previous = cur.fetchone()
    except Exception as e:
        print("SMART ALERT HISTORY WARNING:", e)
    finally:
        if cur:
            cur.close()

    # First assessment: there is no trend to compare yet.
    if not previous:
        return alerts

    previous_prediction = str(previous[1] or "")
    previous_probability = row_high_risk_probability(previous[1], previous[2], previous[3])
    previous_inputs = _load_prediction_input(previous[5])

    # --------------------------------------------------------
    # Model probability trend
    # --------------------------------------------------------
    if previous_probability is not None:
        delta = round(current_probability - previous_probability, 2)
        if delta >= 10:
            alerts.append({
                "title": "Risk Probability Increased",
                "message": (
                    f"Your high-risk class probability changed from "
                    f"{previous_probability:.1f}% to {current_probability:.1f}% "
                    f"({delta:+.1f} percentage points) compared with your previous assessment. "
                    "Review the assessment details and discuss meaningful changes with a qualified healthcare professional."
                ),
                "type": "warning", "priority": "attention", "code": "probability_increase"
            })
        elif delta <= -10:
            alerts.append({
                "title": "Risk Probability Decreased",
                "message": (
                    f"Your high-risk class probability changed from "
                    f"{previous_probability:.1f}% to {current_probability:.1f}% "
                    f"({delta:+.1f} percentage points) compared with your previous assessment. "
                    "This is a model-output change and should not be interpreted as a clinical diagnosis."
                ),
                "type": "info", "priority": "normal", "code": "probability_decrease"
            })

    # --------------------------------------------------------
    # Classification change
    # --------------------------------------------------------
    current_high = "high risk" in str(prediction).lower()
    previous_high = "high risk" in previous_prediction.lower()
    if current_high != previous_high:
        direction = "changed to a higher-risk classification" if current_high else "changed to a lower-risk classification"
        alerts.append({
            "title": "Prediction Classification Changed",
            "message": (
                f"Compared with your previous assessment, the model classification {direction}. "
                f"Current high-risk class probability: {current_probability:.1f}%. "
                "Review both assessments before drawing conclusions."
            ),
            "type": "warning" if current_high else "info",
            "priority": "attention" if current_high else "normal",
            "code": "classification_change"
        })

    # --------------------------------------------------------
    # Meaningful input changes: descriptive only
    # --------------------------------------------------------
    tracked_changes = [
        ("resting_bp", "Resting BP", 20, "mmHg", "increase"),
        ("cholesterol", "Cholesterol", 30, "mg/dL", "increase"),
        ("max_hr", "Maximum heart rate", 20, "bpm", "decrease"),
        ("oldpeak", "ST depression (Oldpeak)", 1.0, "", "increase"),
    ]
    change_lines = []
    for key, label, threshold, unit, direction in tracked_changes:
        old = _safe_float(previous_inputs.get(key))
        new = _safe_float(user_data.get(key))
        if old is None or new is None:
            continue
        diff = new - old
        significant = diff >= threshold if direction == "increase" else diff <= -threshold
        if significant:
            unit_text = f" {unit}" if unit else ""
            change_lines.append(f"{label}: {old:g} → {new:g}{unit_text}")

    if change_lines:
        alerts.append({
            "title": "Assessment Values Changed",
            "message": (
                "Some submitted assessment values changed materially compared with your previous assessment: "
                + "; ".join(change_lines)
                + ". This alert describes data changes only; it does not determine their clinical significance."
            ),
            "type": "info", "priority": "normal", "code": "input_change"
        })

    # Avoid duplicate alerts when the same code already exists for this user
    # very recently. This keeps repeated submissions from flooding the inbox.
    for alert in alerts:
        duplicate = False
        try:
            dcur = mysql.connection.cursor()
            dcur.execute("""
                SELECT id FROM notifications
                WHERE user_id = %s
                  AND title = %s
                  AND created_at >= DATE_SUB(NOW(), INTERVAL 1 DAY)
                LIMIT 1
            """, (user_id, alert["title"]))
            duplicate = dcur.fetchone() is not None
            dcur.close()
        except Exception as e:
            print("SMART ALERT DUPLICATE CHECK WARNING:", e)
        if not duplicate:
            if create_notification(
                user_id,
                alert["title"],
                alert["message"],
                alert["type"],
                reference_type="prediction",
                reference_id=prediction_id,
                priority=alert["priority"]
            ):
                alert["prediction_id"] = prediction_id

    return alerts


# ============================================================
# HEART DISEASE PREDICTION
# ============================================================

@app.route("/heart_prediction", methods=["GET", "POST"])
def heart_prediction():
    if not login_required(): return redirect(url_for("login"))
    if heart_model is None:
        flash("Heart disease model could not be loaded.", "danger"); return redirect(url_for("dashboard"))
    if request.method == "POST":
        try:
            user_data={
                "age":float(request.form["age"]), "sex":request.form["sex"],
                "chest_pain_type":request.form["chest_pain_type"], "resting_bp":float(request.form["resting_bp"]),
                "cholesterol":float(request.form["cholesterol"]), "fasting_bs":int(request.form["fasting_bs"]),
                "resting_ecg":request.form["resting_ecg"], "max_hr":float(request.form["max_hr"]),
                "exercise_angina":request.form["exercise_angina"], "oldpeak":float(request.form["oldpeak"]),
                "st_slope":request.form["st_slope"]
            }
            input_data=pd.DataFrame([[user_data[f] for f in EXPECTED_FIELDS]], columns=MODEL_COLUMNS)
            prediction=int(heart_model.predict(input_data)[0]); probability=heart_model.predict_proba(input_data)[0]
            confidence=round(float(max(probability))*100,2)
            high_risk_probability=round(float(probability[1])*100,2)
            result="High Risk of Heart Disease" if prediction==1 else "Low Risk of Heart Disease"
            risk_level="High Risk" if prediction==1 else "Low Risk"
            risk_factors=generate_risk_factors(user_data)
            xai_explanation=explain_prediction(heart_model,input_data,prediction,MODEL_PATH)
            adaptive_weights=heart_model.adaptive_weights(input_data)[0] if hasattr(heart_model,'adaptive_weights') else {}
            model_details=heart_model.model_details(input_data)[0] if hasattr(heart_model,'model_details') else []
            report_id = request.form.get("report_id")
            extraction_trace = build_extraction_trace(user_data, report_id)
            data_source = extraction_trace["summary"]
            model_analysis={"model_name":"Adaptive Ensemble (Research Model)","prediction":result,"confidence":confidence,"high_risk_probability":high_risk_probability,"description":"The model combines Logistic Regression, Random Forest, Gradient Boosting and SVM. Global validation reliability is combined with sample-level model probability information to adapt the contribution of each base model for the current assessment.","adaptive_weights":adaptive_weights,"model_details":model_details,"probability_label":"Predicted class probability","data_source":data_source}
            # ========================================================
            # SAVE HEART ASSESSMENT -- EXPLICIT, VERIFIED WRITE
            # ========================================================
            save_user_id = session.get("user_id")
            if save_user_id is None and session.get("email"):
                ucur = mysql.connection.cursor()
                ucur.execute("SELECT id FROM users WHERE email=%s LIMIT 1", (session["email"],))
                urow = ucur.fetchone()
                ucur.close()
                if urow:
                    save_user_id = urow[0]
                    session["user_id"] = save_user_id
            if save_user_id is None:
                raise RuntimeError("Unable to resolve the logged-in user before saving the Heart Disease assessment.")

            cur = mysql.connection.cursor()
            try:
                cur.execute(
                    """INSERT INTO predictions
                       (user_id, disease_name, prediction, confidence, high_risk_probability, input_data)
                       VALUES (%s,%s,%s,%s,%s,%s)""",
                    (save_user_id, "Heart Disease", result, confidence,
                     high_risk_probability, json.dumps(user_data))
                )
                prediction_id = cur.lastrowid
                mysql.connection.commit()

                # Verify the record exists immediately after commit.
                cur.execute(
                    "SELECT id FROM predictions WHERE id=%s AND user_id=%s LIMIT 1",
                    (prediction_id, save_user_id)
                )
                if not cur.fetchone():
                    raise RuntimeError("Heart Disease assessment was not found after database commit.")
                _phase16_write_audit(prediction_id, save_user_id, "Heart Disease", result, high_risk_probability, user_data)
            except Exception:
                try:
                    mysql.connection.rollback()
                except Exception:
                    pass
                raise
            finally:
                cur.close()
            create_notification(
                session["user_id"],
                "High Risk Prediction" if prediction == 1 else "Prediction Completed",
                "Your latest heart disease risk prediction indicates a higher predicted risk. Please review your results and discuss concerns with a qualified healthcare professional." if prediction == 1 else "Your heart disease risk prediction has been completed. You can review the result and recommendations.",
                "danger" if prediction == 1 else "success",
                reference_type="prediction", reference_id=prediction_id,
                priority="important" if prediction == 1 else "normal"
            )
            smart_alerts=generate_smart_health_alerts(
                session["user_id"], prediction_id, user_data, result, probability, report_id
            )
            recommendations=generate_recommendations(user_data,prediction)
            # Retrieve recent saved assessments for personalized monitoring context.
            recent_history=[]
            try:
                hcur=mysql.connection.cursor()
                hcur.execute("""SELECT prediction, confidence, high_risk_probability, prediction_date FROM predictions WHERE user_id=%s AND id<>%s ORDER BY prediction_date DESC LIMIT 5""", (session["user_id"], prediction_id))
                for row in hcur.fetchall():
                    recent_history.append({"prediction": row[0], "confidence": float(row[1] or 0), "high_risk_probability": row_high_risk_probability(row[0], row[1], row[2]), "prediction_date": row[3].strftime("%d %b %Y") if hasattr(row[3], "strftime") else str(row[3])})
                hcur.close()
            except Exception as history_error:
                print("PERSONALIZED HISTORY WARNING:", history_error)
                recent_history=[]
            personalized_plan=generate_personalized_plan(user_data,prediction,recent_history,xai_explanation)
            ai_explanation=generate_ai_explanation(user_data,prediction,result,confidence,risk_factors)
            prediction_date=datetime.now().strftime("%d %B %Y, %I:%M %p")
            return render_template("result.html",result=result,confidence=confidence,high_risk_probability=high_risk_probability,recommendations=recommendations,risk_factors=risk_factors,xai_explanation=xai_explanation,ai_explanation=ai_explanation,model_analysis=model_analysis,user_data=user_data,prediction_id=prediction_id,prediction_date=prediction_date,risk_level=risk_level,data_source=data_source,report_id=report_id,extraction_trace=extraction_trace,personalized_plan=personalized_plan,recent_history=recent_history,smart_alerts=smart_alerts)
        except (KeyError,ValueError,TypeError) as e:
            print("PREDICTION INPUT ERROR:",e); flash(f"Please check all assessment fields: {e}","danger"); return redirect(url_for("heart_prediction"))
        except Exception as e:
            print("PREDICTION ERROR:",e)
            try: mysql.connection.rollback()
            except: pass
            flash(f"Unable to generate prediction: {e}","danger"); return redirect(url_for("heart_prediction"))
    return render_template("prediction.html", extracted_data=None, validation=None, compatibility=None, report_type=None, extracted_count=0, missing_count=len(EXPECTED_FIELDS), report_id=None)


# ============================================================
# DIABETES MODULE HELPERS
# ============================================================

DIABETES_DISPLAY_FIELDS = [
    ("pregnancies", "Pregnancies", ""),
    ("glucose", "Glucose", "mg/dL"),
    ("blood_pressure", "Blood Pressure", "mmHg"),
    ("skin_thickness", "Skin Thickness", "mm"),
    ("insulin", "Insulin", "µU/mL"),
    ("bmi", "BMI", "kg/m²"),
    ("diabetes_pedigree_function", "Diabetes Pedigree Function", ""),
    ("age", "Age", "years"),
]


def diabetes_risk_factors(data):
    factors = []
    # Project/model input signals only; these are not clinical diagnostic thresholds.
    if float(data.get("glucose", 0) or 0) >= 126:
        factors.append({"title": "Glucose input", "message": f"Glucose is recorded as {data.get('glucose')} mg/dL and is one of the model inputs associated with this assessment."})
    if float(data.get("bmi", 0) or 0) >= 30:
        factors.append({"title": "BMI input", "message": f"BMI is recorded as {data.get('bmi')} and contributes to the model's input pattern."})
    if float(data.get("age", 0) or 0) >= 45:
        factors.append({"title": "Age input", "message": f"Age ({data.get('age')} years) is included in the model input pattern."})
    if float(data.get("blood_pressure", 0) or 0) >= 140:
        factors.append({"title": "Blood pressure input", "message": f"Blood pressure input is recorded as {data.get('blood_pressure')} mmHg."})
    if not factors:
        factors.append({"title": "No project rule-based signal", "message": "No predefined project rule-based attention signal was triggered by the submitted values."})
    return factors


def diabetes_recommendations(data, prediction):
    medical = [
        "Review this machine-learning estimate together with your actual measurements and medical history with a qualified healthcare professional.",
        "If you have symptoms or concerns about glucose or metabolic health, discuss appropriate clinical testing with a healthcare professional.",
    ]
    lifestyle = [
        "Prefer regular physical activity appropriate for your health status and professional advice.",
        "Avoid smoking and limit highly processed foods and excess added sugar.",
    ]
    diet = [
        "Choose a balanced eating pattern with vegetables, whole grains, legumes and appropriate protein sources.",
        "Prefer water or unsweetened beverages and moderate highly processed foods.",
    ]
    follow = [
        "Keep track of relevant measurements recommended by your healthcare professional.",
        "Do not start, stop or change medication based on this model output.",
    ]
    return {"Medical Advice": medical, "Lifestyle": lifestyle, "Diet": diet, "Follow-up": follow}


def diabetes_personalized_plan(data, prediction, recent_history, xai):
    priorities=[]
    for f in diabetes_risk_factors(data):
        if f["title"] == "No project rule-based signal":
            continue
        key = f["title"].replace(" input", "")
        val = data.get({"Glucose":"glucose","BMI":"bmi","Age":"age","Blood pressure":"blood_pressure"}.get(key, ""), "")
        priorities.append({"field": f["title"], "level": "Attention", "value": val, "reason": f["message"], "action": "Use this information as part of professional interpretation rather than as a standalone diagnosis."})
    if not priorities:
        priorities.append({"field":"Assessment pattern","level":"Informational","value":"No rule-based signal","reason":"No predefined project attention signal was triggered.","action":"Continue reviewing the assessment with the rest of your health information."})
    monitoring=[
        f"Current assessment is classified by the model as {'higher' if prediction else 'lower'} risk; this is a model output, not a diagnosis.",
        f"This is assessment #{len(recent_history)+1} in the retrieved recent history context.",
    ]
    if len(recent_history)>=1:
        prev=recent_history[0].get('high_risk_probability')
        if prev is not None:
            monitoring.append(f"Previous stored diabetes high-risk probability: {prev:.2f}%. Changes represent model-output changes, not proof of a clinical change.")
    if xai and xai.get('available'):
        monitoring.append(f"The current assessment has {len(xai.get('features', []))} local XAI attribution(s) describing model behavior.")
    return {
        "priorities": priorities,
        "actions":["Maintain balanced daily habits and follow professional advice appropriate to your health status.","Use the model result as an educational prompt rather than a diagnosis."],
        "monitoring":monitoring,
        "disclaimer":"These recommendations are transparent, rule-based educational guidance derived from submitted assessment data and stored model history. They are not a diagnosis, prescription, or individualized treatment plan."
    }


def early_diabetes_risk_factors(data):
    factors = []
    labels = {
        "polyuria": "Frequent urination (Polyuria)",
        "polydipsia": "Excessive thirst (Polydipsia)",
        "sudden_weight_loss": "Sudden weight loss",
        "polyphagia": "Increased hunger (Polyphagia)",
        "visual_blurring": "Visual blurring",
        "weakness": "Weakness",
        "delayed_healing": "Delayed healing",
        "partial_paresis": "Partial paresis",
        "obesity": "Obesity",
    }
    for key, label in labels.items():
        if str(data.get(key, "")).strip().lower() == "yes":
            factors.append({"title": label, "message": f"{label} was recorded as present in this assessment and is one of the model inputs."})
    if str(data.get("gender", "")).strip().lower() in {"male", "female"} and float(data.get("age", 0) or 0) >= 45:
        factors.append({"title": "Age input", "message": f"Age ({data.get('age')} years) is included in the Early Diabetes model input pattern."})
    if not factors:
        factors.append({"title": "No project rule-based signal", "message": "No predefined project rule-based attention signal was triggered by the submitted Early Diabetes values."})
    return factors


def early_diabetes_recommendations(data, prediction):
    return {
        "Medical Advice": [
            "Review this machine-learning estimate with a qualified healthcare professional and consider appropriate clinical testing when indicated.",
            "Do not use this model output as a diagnosis or as a basis for changing medication.",
        ],
        "Lifestyle": [
            "Maintain regular physical activity appropriate for your health status and professional advice.",
            "Avoid smoking and limit highly processed foods and excess added sugar.",
        ],
        "Follow-up": [
            "Keep track of relevant symptoms and measurements recommended by your healthcare professional.",
            "Discuss persistent symptoms such as excessive thirst, frequent urination or unexplained weight change with a professional.",
        ]
    }


def build_diabetes_extraction_trace(user_data, report_id):
    trace={"fields":{},"summary":"Manual assessment entry"}
    if not report_id:
        for key,label,_ in DIABETES_DISPLAY_FIELDS:
            trace["fields"][key]={"status":"manual","label":"Manual entry","final":user_data.get(key)}
        return trace
    cur=None
    try:
        cur=mysql.connection.cursor()
        cur.execute("SELECT extracted_data FROM medical_reports WHERE id=%s AND user_id=%s LIMIT 1", (int(report_id), session["user_id"]))
        row=cur.fetchone()
        extracted=json.loads(row[0]) if row and row[0] else {}
        changed=0; extracted_count=0
        for key,label,_ in DIABETES_DISPLAY_FIELDS:
            original=extracted.get(key)
            final=user_data.get(key)
            if original is not None:
                extracted_count+=1
                status="verified" if str(original)==str(final) else "edited"
                if status=="edited": changed+=1
                trace["fields"][key]={"status":status,"label":"Extracted & verified" if status=="verified" else "Extracted & modified","extracted":original,"final":final}
            else:
                trace["fields"][key]={"status":"manual","label":"Manual entry","final":final}
        if changed: trace["summary"]=f"Medical report extraction + user verification ({changed} field(s) modified)"
        elif extracted_count: trace["summary"]="Medical report extraction + user verification"
        else: trace["summary"]="Manual assessment entry"
    except Exception as e:
        print("DIABETES TRACE WARNING:",e)
        for key,label,_ in DIABETES_DISPLAY_FIELDS:
            trace["fields"][key]={"status":"manual","label":"Manual entry","final":user_data.get(key)}
    finally:
        if cur: cur.close()
    return trace


def generate_diabetes_smart_alerts(user_id, prediction_id, current_probability):
    alerts=[]; cur=None
    try:
        cur=mysql.connection.cursor()
        cur.execute("""SELECT id,prediction,confidence,high_risk_probability,prediction_date FROM predictions WHERE user_id=%s AND disease_name='Diabetes' AND id<>%s ORDER BY prediction_date DESC,id DESC LIMIT 1""", (user_id,prediction_id))
        row=cur.fetchone()
        if not row: return alerts
        prev_high=row[3]
        if prev_high is None:
            prev_high=float(row[2] or 0) if "high" in str(row[1]).lower() else 100-float(row[2] or 0)
        delta=round(float(current_probability)-float(prev_high),2)
        if abs(delta)>=5:
            alerts.append({"title":"Diabetes High-Risk Probability Changed","priority":"attention" if abs(delta)>=10 else "normal","message":f"Your diabetes high-risk probability changed from {float(prev_high):.2f}% to {float(current_probability):.2f}% ({delta:+.2f} percentage points) compared with your previous diabetes assessment. This is a model-output change, not a clinical diagnosis."})
        current_class="high" if current_probability>=50 else "low"
        previous_class="high" if float(prev_high)>=50 else "low"
        if current_class!=previous_class:
            alerts.append({"title":"Diabetes Prediction Classification Changed","priority":"important","message":f"Compared with your previous diabetes assessment, the model classification changed to {'higher-risk' if current_class=='high' else 'lower-risk'}. Current high-risk probability: {float(current_probability):.2f}%. Review both assessments before drawing conclusions."})
    except Exception as e: print("DIABETES SMART ALERT ERROR:",e)
    finally:
        if cur: cur.close()
    return alerts


# ============================================================
# DIABETES PREDICTION
# ============================================================

@app.route("/diabetes_prediction", methods=["GET", "POST"])
def diabetes_prediction():
    if not login_required():
        return redirect(url_for("login"))
    if diabetes_engine is None:
        flash("Diabetes model could not be loaded. Please restart the application and check the diabetes dataset.", "danger")
        return redirect(url_for("dashboard"))
    if request.method == "GET":
        return render_template("diabetes_prediction.html", extracted_data=None, report_type=None, extracted_count=0, missing_count=len(DIABETES_FIELDS), report_id=None)
    try:
        user_data={}
        for f in DIABETES_FIELDS:
            raw=request.form.get(f, "").strip()
            if raw=="": raise ValueError(f"Missing value for {f.replace('_',' ')}")
            user_data[f]=float(raw)
        result_data=diabetes_engine.predict(user_data)
        prediction=int(result_data["prediction"])
        high_prob=float(result_data["high_risk_probability"])
        confidence=float(result_data["predicted_class_probability"])
        result="Higher Risk of Diabetes" if prediction==1 else "Lower Risk of Diabetes"
        risk_level="High Risk" if prediction==1 else "Low Risk"
        factors=diabetes_risk_factors(user_data)
        xai=diabetes_engine.shap_explanation(user_data)
        details=result_data["model_details"]
        report_id=request.form.get("report_id")
        trace=build_diabetes_extraction_trace(user_data, report_id)
        recent=[]; cur=None
        try:
            cur=mysql.connection.cursor()
            cur.execute("SELECT prediction,confidence,high_risk_probability,prediction_date FROM predictions WHERE user_id=%s AND disease_name='Diabetes' ORDER BY prediction_date DESC,id DESC LIMIT 5", (session["user_id"],))
            for r in cur.fetchall():
                hp=row_high_risk_probability(r[0],r[1],r[2])
                recent.append({"prediction":r[0],"confidence":float(r[1] or 0),"high_risk_probability":float(hp),"prediction_date":r[3].strftime("%d %b %Y") if hasattr(r[3],"strftime") else str(r[3])})
        finally:
            if cur: cur.close()
        model_analysis={"model_name":"Adaptive Weighted Ensemble (Diabetes)","prediction":result,"confidence":confidence,"high_risk_probability":high_prob,"description":"The diabetes module combines five calibrated base classifiers. Global cross-validation balanced accuracy is used to initialize model weights, with sample-level confidence adjustment for the current assessment.","model_details":details,"probability_label":"Predicted class probability","data_source":trace["summary"]}
        cur=mysql.connection.cursor()
        cur.execute("INSERT INTO predictions (user_id,disease_name,prediction,confidence,high_risk_probability,input_data) VALUES (%s,%s,%s,%s,%s,%s)",(session["user_id"],"Diabetes",result,confidence,high_prob,json.dumps(user_data)))
        mysql.connection.commit(); prediction_id=cur.lastrowid; cur.close()
        _phase16_write_audit(prediction_id, session["user_id"], "Diabetes", result, high_prob, user_data)
        create_notification(session["user_id"], "Diabetes Prediction Completed", f"Your diabetes risk assessment has been completed. Predicted high-risk probability: {high_prob:.2f}%. Review the report for model details.", "danger" if prediction else "success", reference_type="prediction", reference_id=prediction_id, priority="important" if prediction else "normal")
        smart_alerts=generate_diabetes_smart_alerts(session["user_id"],prediction_id,high_prob)
        for alert in smart_alerts:
            create_notification(session["user_id"], alert["title"], alert["message"], "warning", reference_type="prediction", reference_id=prediction_id, priority=alert["priority"])
        personalized=diabetes_personalized_plan(user_data,prediction,recent,xai)
        recommendations=diabetes_recommendations(user_data,prediction)
        ai_explanation=(f"The machine-learning model produced a {'higher' if prediction else 'lower'} predicted risk classification for diabetes with a predicted-class probability of {confidence:.1f}%. The high-risk class probability is {high_prob:.1f}%. This is a model-based estimate, not a medical diagnosis.")
        return render_template("diabetes_result.html", disease_name="Diabetes", result=result, confidence=confidence, high_risk_probability=high_prob, risk_level=risk_level, user_data=user_data, prediction_id=prediction_id, prediction_date=datetime.now().strftime("%d %B %Y, %I:%M %p"), data_source=trace["summary"], extraction_trace=trace, risk_factors=factors, model_analysis=model_analysis, xai_explanation=xai, ai_explanation=ai_explanation, personalized_plan=personalized, recent_history=recent, smart_alerts=smart_alerts, recommendations=recommendations)
    except Exception as e:
        print("DIABETES PREDICTION ERROR:",e)
        try: mysql.connection.rollback()
        except: pass
        flash(f"Unable to generate diabetes prediction: {e}","danger")
        return redirect(url_for("diabetes_prediction"))


@app.route("/early_diabetes_prediction", methods=["GET", "POST"])
def early_diabetes_prediction():
    """Restored Early Diabetes assessment from the original Diabetes_model.zip.

    This route is additive: the current /diabetes_prediction route and all
    existing Diabetes analytics remain unchanged.
    """
    if not login_required():
        return redirect(url_for("login"))
    if early_diabetes_engine is None:
        flash("Early Diabetes model could not be loaded. Please restart the application and check the supplied dataset.", "danger")
        return redirect(url_for("dashboard"))

    if request.method == "GET":
        return render_template(
        "early_diabetes_prediction.html",
        extracted_data={},
        extracted_count=0,
        missing_count=16,
        report_type=None,
        report_id=None
        )

    try:
        user_data = {}
        for field in EARLY_FIELDS:
            value = request.form.get(field, "").strip()
            if not value:
                raise ValueError(f"Missing value for {field.replace('_', ' ')}")
            user_data[field] = value
        user_data["age"] = float(user_data["age"])
        if user_data["gender"].lower() not in {"male", "female"}:
            raise ValueError("Gender must be Male or Female")
        for field in EARLY_FIELDS[2:]:
            if user_data[field].lower() not in {"yes", "no"}:
                raise ValueError(f"{field.replace('_', ' ')} must be Yes or No")

        result_data = early_diabetes_engine.predict(user_data)
        prediction = int(result_data["prediction"])
        high_prob = float(result_data["high_risk_probability"])
        confidence = float(result_data["predicted_class_probability"])
        result = "Higher Risk of Early Diabetes" if prediction else "Lower Risk of Early Diabetes"
        risk_level = "High Risk" if prediction else "Low Risk"
        factors = early_diabetes_risk_factors(user_data)
        xai = early_diabetes_engine.shap_explanation(user_data)
        details = result_data["model_details"]
        recommendations = early_diabetes_recommendations(user_data, prediction)
        model_analysis = {
            "model_name": "Adaptive Weighted Ensemble (Early Diabetes)",
            "prediction": result,
            "confidence": confidence,
            "high_risk_probability": high_prob,
            "description": "The restored Early Diabetes module combines five base classifiers. Cross-validation balanced accuracy initializes the global weights, followed by sample-level confidence adjustment for the current assessment.",
            "model_details": details,
            "probability_label": "Predicted class probability",
        }

        cur = mysql.connection.cursor()
        cur.execute(
            "INSERT INTO predictions (user_id,disease_name,prediction,confidence,high_risk_probability,input_data) VALUES (%s,%s,%s,%s,%s,%s)",
            (session["user_id"], "Early Diabetes", result, confidence, high_prob, json.dumps(user_data))
        )
        mysql.connection.commit()
        prediction_id = cur.lastrowid
        cur.close()
        _phase16_write_audit(prediction_id, session["user_id"], "Early Diabetes", result, high_prob, user_data)

        create_notification(
            session["user_id"],
            "Early Diabetes Prediction Completed",
            f"Your Early Diabetes risk assessment has been completed. Predicted high-risk probability: {high_prob:.2f}%.",
            "danger" if prediction else "success",
            reference_type="prediction", reference_id=prediction_id,
            priority="important" if prediction else "normal"
        )

        return render_template(
            "early_diabetes_result.html",
            disease_name="Early Diabetes",
            result=result,
            confidence=confidence,
            high_risk_probability=high_prob,
            risk_level=risk_level,
            user_data=user_data,
            prediction_id=prediction_id,
            prediction_date=datetime.now().strftime("%d %B %Y, %I:%M %p"),
            risk_factors=factors,
            model_analysis=model_analysis,
            xai_explanation=xai,
            recommendations=recommendations,
        )
    except Exception as e:
        print("EARLY DIABETES PREDICTION ERROR:", e)
        try:
            mysql.connection.rollback()
        except Exception:
            pass
        flash(f"Unable to generate Early Diabetes prediction: {e}", "danger")
        return redirect(url_for("early_diabetes_prediction"))


@app.route("/upload_diabetes_report", methods=["POST"])
def upload_diabetes_report():
    if not login_required(): return redirect(url_for("login"))
    if "report" not in request.files or not request.files["report"].filename:
        flash("Please select a diabetes medical report.","warning"); return redirect(url_for("diabetes_prediction"))
    file=request.files["report"]
    if not file.filename.lower().endswith(".pdf"):
        flash("Only PDF reports are supported.","danger"); return redirect(url_for("diabetes_prediction"))
    filepath=None; cur=None
    try:
        original=secure_filename(file.filename); filename=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{original}"; filepath=os.path.join(app.config["UPLOAD_FOLDER"],filename); file.save(filepath)
        text=extract_text(filepath)
        if not text or len(text.strip())<20: raise ValueError("Unable to read meaningful text from the PDF.")
        report_type=detect_report_type(text)
        extracted=parse_diabetes_report(text)
        count=sum(v is not None for v in extracted.values()); missing=len(DIABETES_FIELDS)-count
        cur=mysql.connection.cursor(); status="Complete" if missing==0 else "Incomplete"
        cur.execute("INSERT INTO medical_reports (user_id,report_name,report_type,file_path,extraction_status,extracted_fields,missing_fields,extracted_data) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",(session["user_id"],original,"diabetes" if report_type in ("diabetes","general") else report_type,filepath,status,count,missing,json.dumps(extracted))); mysql.connection.commit(); report_id=cur.lastrowid; cur.close(); cur=None
        create_notification(session["user_id"],"Diabetes Medical Report Analyzed",f"{count} of {len(DIABETES_FIELDS)} diabetes model inputs were extracted. {missing} still require manual entry.","success" if missing==0 else "warning",reference_type="report",reference_id=report_id,priority="normal" if missing==0 else "attention")
        return render_template("diabetes_prediction.html", extracted_data=extracted, report_type=report_type, extracted_count=count, missing_count=missing, report_id=report_id)
    except Exception as e:
        if cur:
            try: mysql.connection.rollback(); cur.close()
            except: pass
        if filepath and os.path.exists(filepath):
            # Keep valid uploaded reports for the Reports center; only remove on unreadable PDFs.
            pass
        print("DIABETES REPORT ERROR:",e); flash(f"Unable to process diabetes medical report: {e}","danger"); return redirect(url_for("diabetes_prediction"))

# ============================================================
# EARLY DIABETES MEDICAL REPORT UPLOAD
# ============================================================

@app.route(
    "/early_diabetes_report_upload",
    methods=["POST"]
)
def early_diabetes_report_upload():

    if not login_required():
        return redirect(url_for("login"))

    if "early_diabetes_report" not in request.files:

        flash(
            "Please select an Early Diabetes medical report.",
            "danger"
        )

        return redirect(
            url_for("early_diabetes_prediction")
        )

    file = request.files[
        "early_diabetes_report"
    ]

    if not file or file.filename == "":

        flash(
            "Please select a PDF medical report.",
            "warning"
        )

        return redirect(
            url_for("early_diabetes_prediction")
        )

    if not file.filename.lower().endswith(".pdf"):

        flash(
            "Only PDF medical reports are supported.",
            "danger"
        )

        return redirect(
            url_for("early_diabetes_prediction")
        )

    filepath = None

    try:

        # ----------------------------------------------------
        # SAVE FILE
        # ----------------------------------------------------

        original_filename = secure_filename(
            file.filename
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            f"early_diabetes_"
            f"{timestamp}_"
            f"{original_filename}"
        )

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        # ----------------------------------------------------
        # EXTRACT PDF TEXT
        # ----------------------------------------------------

        text = extract_text(
            filepath
        )

        if (
            not text
            or len(text.strip()) < 20
        ):

            if os.path.exists(filepath):
                os.remove(filepath)

            flash(
                "Unable to read meaningful text from "
                "the uploaded PDF.",
                "danger"
            )

            return redirect(
                url_for("early_diabetes_prediction")
            )

        # ----------------------------------------------------
        # PARSE EARLY DIABETES DATA
        # ----------------------------------------------------

        extracted_data = (
            parse_early_diabetes_report(text)
        )

        extracted_data = (
            ensure_early_diabetes_fields(
                extracted_data
            )
        )

        # ----------------------------------------------------
        # COUNT
        # ----------------------------------------------------

        (
            extracted_count,
            missing_count
        ) = count_early_diabetes_fields(
            extracted_data
        )

        # ----------------------------------------------------
        # REPORT TYPE
        # ----------------------------------------------------

        report_type = (
            "Early Diabetes / Symptom Report"
        )

        # ----------------------------------------------------
        # SAVE REPORT INFORMATION
        # ----------------------------------------------------

        report_id = None

        try:

            cur = mysql.connection.cursor()

            extraction_status = (
                "Complete"
                if missing_count == 0
                else "Incomplete"
            )

            extracted_fields = ",".join(
                field
                for field in EARLY_DIABETES_FIELDS
                if extracted_data.get(field)
                not in (None, "")
            )

            missing_fields = ",".join(
                field
                for field in EARLY_DIABETES_FIELDS
                if extracted_data.get(field)
                in (None, "")
            )

            cur.execute(
                """
                INSERT INTO medical_reports
                (
                    user_id,
                    report_name,
                    report_type,
                    file_path,
                    extraction_status,
                    extracted_fields,
                    missing_fields
                )
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    session["user_id"],
                    original_filename,
                    report_type,
                    filepath,
                    extraction_status,
                    extracted_fields,
                    missing_fields
                )
            )

            mysql.connection.commit()

            report_id = cur.lastrowid

            cur.close()

        except Exception as db_error:

            mysql.connection.rollback()

            print(
                "EARLY DIABETES REPORT DB ERROR:",
                db_error
            )

            # Extraction should still work even if
            # report-history storage has an issue.

        # ----------------------------------------------------
        # USER MESSAGE
        # ----------------------------------------------------

        if extracted_count == 0:

            flash(
                "The PDF was readable, but no supported "
                "Early Diabetes parameters were detected. "
                "Please enter the values manually.",
                "warning"
            )

        elif missing_count > 0:

            flash(
                f"{extracted_count} of "
                f"{len(EARLY_DIABETES_FIELDS)} "
                f"Early Diabetes parameters were extracted. "
                f"Please enter the remaining "
                f"{missing_count} manually.",
                "warning"
            )

        else:

            flash(
                "All Early Diabetes parameters were "
                "extracted. Please verify them before "
                "generating the assessment.",
                "success"
            )

        # ----------------------------------------------------
        # DEBUG
        # ----------------------------------------------------

        print("=" * 70)
        print("EARLY DIABETES REPORT ANALYSIS")
        print("=" * 70)

        print(
            "Report:",
            original_filename
        )

        print(
            "Extracted:",
            extracted_count
        )

        print(
            "Missing:",
            missing_count
        )

        print(
            "Data:",
            extracted_data
        )

        print("=" * 70)

        # ----------------------------------------------------
        # RETURN TO EARLY DIABETES FORM
        # ----------------------------------------------------

        return render_template(
            "early_diabetes_prediction.html",
            extracted_data=extracted_data,
            extracted_count=extracted_count,
            missing_count=missing_count,
            report_type=report_type,
            report_id=report_id
        )

    except Exception as e:

        print("=" * 70)
        print("EARLY DIABETES REPORT ERROR")
        print(e)
        print("=" * 70)

        if filepath and os.path.exists(filepath):

            try:
                os.remove(filepath)
            except Exception:
                pass

        flash(
            f"Unable to process the medical report: {e}",
            "danger"
        )

        return redirect(
            url_for("early_diabetes_prediction")
        )
# ============================================================
# PHASE 8 - RESEARCH EVALUATION
# ============================================================

@app.route("/research_evaluation")
def research_evaluation():
    if not login_required():
        return redirect(url_for("login"))
    try:
        path = os.path.join(BASE_DIR, "phase8_research_evaluation.json")
        if not os.path.exists(path):
            flash("Phase 8 research evaluation report is not available yet. Run algorithms/build_phase8_evaluation.py first.", "warning")
            return redirect(url_for("research_analytics"))
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        models = report.get("models", {})
        labels = {"lr":"Logistic Regression","rf":"Random Forest","gb":"Gradient Boosting","svm":"SVM","static":"Static Probability Average","adaptive":"Adaptive Ensemble"}
        rows = []
        for key in ["lr","rf","gb","svm","static","adaptive"]:
            m = models.get(key, {})
            rows.append({"key":key,"name":labels[key],"metrics":m})
        best_auc = max(rows, key=lambda r: r["metrics"].get("roc_auc",{}).get("mean",-1))
        best_brier = min(rows, key=lambda r: r["metrics"].get("brier",{}).get("mean",999))
        return render_template("research_evaluation.html", report=report, rows=rows, best_auc=best_auc, best_brier=best_brier)
    except Exception as e:
        print("PHASE 8 RESEARCH EVALUATION ERROR:", e)
        flash("Unable to load Phase 8 research evaluation.", "danger")
        return redirect(url_for("research_analytics"))


# ============================================================
# RESEARCH ANALYTICS
# ============================================================

@app.route("/research_analytics")
def research_analytics():
    if not login_required():
        return redirect(url_for("login"))
    try:
        report_path = os.path.join(BASE_DIR, "research_analytics.json")
        if not os.path.exists(report_path):
            flash("Research analytics report is not available yet. Run the analytics builder first.", "warning")
            return redirect(url_for("dashboard"))
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        validation_path = os.path.join(BASE_DIR, "research_validation_report.json")
        if os.path.exists(validation_path):
            try:
                with open(validation_path, "r", encoding="utf-8") as vf:
                    validation = json.load(vf)
                report["five_fold_cv_roc_auc"] = validation.get("five_fold_cv_roc_auc", {})
            except Exception:
                report["five_fold_cv_roc_auc"] = {}
        else:
            report["five_fold_cv_roc_auc"] = {}
        models = dict(report.get("individual_models", {}))
        models["Static Probability Average"] = report.get("static_probability_average", {})
        models["Adaptive Ensemble"] = report.get("adaptive_ensemble", {})
        best_accuracy = max(models.items(), key=lambda x: x[1].get("accuracy", -1))
        best_f1 = max(models.items(), key=lambda x: x[1].get("f1", -1))
        best_auc = max(models.items(), key=lambda x: x[1].get("roc_auc", -1))
        best_recall = max(models.items(), key=lambda x: x[1].get("recall", -1))
        return render_template("research_analytics.html", report=report, models=models,
            best_accuracy=best_accuracy, best_f1=best_f1, best_auc=best_auc, best_recall=best_recall)
    except Exception as e:
        print("RESEARCH ANALYTICS ERROR:", e)
        flash("Unable to load research analytics.", "danger")
        return redirect(url_for("dashboard"))


# ============================================================
# LONGITUDINAL HEALTH PROFILE
# ============================================================

@app.route("/health-profile")
def health_profile():

    if not login_required():
        return redirect(url_for("login"))

    cur = None
    try:
        cur = mysql.connection.cursor()

        # --------------------------------------------------------
        # User summary
        # --------------------------------------------------------
        cur.execute("SELECT id, fullname, email FROM users WHERE id = %s LIMIT 1", (session["user_id"],))
        user_row = cur.fetchone()
        fullname = user_row[1] if user_row else session.get("fullname", "User")
        email = user_row[2] if user_row else session.get("email", "")

        # --------------------------------------------------------
        # Complete longitudinal prediction history
        # --------------------------------------------------------
        cur.execute("""
            SELECT id, disease_name, prediction, confidence,
                   high_risk_probability, input_data, prediction_date
            FROM predictions
            WHERE user_id = %s
            ORDER BY prediction_date ASC, id ASC
        """, (session["user_id"],))
        rows = cur.fetchall()

        timeline = []
        for row in rows:
            prediction_id, disease_name, prediction, confidence, stored_high, raw_input, prediction_date = row
            high_prob = row_high_risk_probability(prediction, confidence, stored_high)
            inputs = _load_prediction_input(raw_input)
            timeline.append({
                "id": prediction_id,
                "disease": disease_name or "Unknown",
                "prediction": prediction or "Unknown",
                "confidence": round(float(confidence or 0), 2),
                "high_risk_probability": round(float(high_prob), 2),
                "date": prediction_date.strftime("%d %b %Y") if prediction_date else "N/A",
                "time": prediction_date.strftime("%I:%M %p") if prediction_date else "",
                "timestamp": prediction_date.isoformat() if prediction_date else "",
                "inputs": inputs,
            })

        # --------------------------------------------------------
        # Summary statistics
        # --------------------------------------------------------
        total_assessments = len(timeline)
        high_count = sum(1 for x in timeline if "high" in x["prediction"].lower())
        low_count = sum(1 for x in timeline if "high" not in x["prediction"].lower())

        # --------------------------------------------------------
# HEART DISEASE RECORDS
# Accept all supported Heart Disease naming variants.
# --------------------------------------------------------
        heart_rows = [
            x for x in timeline
            if str(x.get("disease", "")).strip().lower()
            in ("heart disease", "heart", "heart_disease")
        ]
        heart_high_probs = [x["high_risk_probability"] for x in heart_rows]
        avg_high_probability = round(sum(heart_high_probs) / len(heart_high_probs), 2) if heart_high_probs else None

# --------------------------------------------------------
# DIABETES RECORDS
# --------------------------------------------------------
        diabetes_rows_for_summary = [
            x for x in timeline
            if str(x.get("disease", "")).strip().lower() == "diabetes"
        ]
        diabetes_high_probs = [x["high_risk_probability"] for x in diabetes_rows_for_summary]
        avg_diabetes_high_probability = round(sum(diabetes_high_probs) / len(diabetes_high_probs), 2) if diabetes_high_probs else None

        latest = timeline[-1] if timeline else None
        first = timeline[0] if timeline else None
        previous = timeline[-2] if len(timeline) >= 2 else None

        latest_change = None
        if latest and previous:
            latest_change = round(
                latest["high_risk_probability"] - previous["high_risk_probability"], 2
            )

        # --------------------------------------------------------
        # Previous vs current Heart Disease input comparison
        # --------------------------------------------------------
        comparison = []
        if latest and previous and latest["disease"].lower() == previous["disease"].lower():
            if latest["disease"].lower() == "heart disease":
                fields = [
                    ("age", "Age", "years"), ("sex", "Sex", ""), ("chest_pain_type", "Chest Pain Type", ""),
                    ("resting_bp", "Resting Blood Pressure", "mmHg"), ("cholesterol", "Cholesterol", "mg/dL"),
                    ("fasting_bs", "Fasting Blood Sugar", ""), ("resting_ecg", "Resting ECG", ""),
                    ("max_hr", "Maximum Heart Rate", "bpm"), ("exercise_angina", "Exercise Angina", ""),
                    ("oldpeak", "Oldpeak", ""), ("st_slope", "ST Slope", ""),
                ]
            elif latest["disease"].lower() == "diabetes":
                fields = [
                    ("pregnancies", "Pregnancies", ""), ("glucose", "Glucose", "mg/dL"),
                    ("blood_pressure", "Blood Pressure", "mmHg"), ("skin_thickness", "Skin Thickness", "mm"),
                    ("insulin", "Insulin", "µU/mL"), ("bmi", "BMI", "kg/m²"),
                    ("diabetes_pedigree_function", "Diabetes Pedigree Function", ""), ("age", "Age", "years"),
                ]
            else:
                fields = []
            for key, label, unit in fields:
                old = previous["inputs"].get(key, "")
                new = latest["inputs"].get(key, "")
                if str(old) == str(new):
                    status = "unchanged"
                elif old in (None, "", "N/A"):
                    status = "added"
                elif new in (None, "", "N/A"):
                    status = "missing"
                else:
                    status = "changed"
                comparison.append({"key": key, "label": label, "old": old, "new": new, "unit": unit, "status": status})

        # --------------------------------------------------------
        # Chart data: Heart Disease high-risk probability
        # --------------------------------------------------------
        chart_rows = heart_rows[-20:]
        chart_labels = [x["date"] for x in chart_rows]
        chart_values = [x["high_risk_probability"] for x in chart_rows]
        chart_predictions = [x["prediction"] for x in chart_rows]

        diabetes_rows = [
            x for x in timeline
            if str(x.get("disease", "")).strip().lower() == "diabetes"
        ]
        diabetes_chart_rows = diabetes_rows[-20:]
        diabetes_chart_labels = [x["date"] for x in diabetes_chart_rows]
        diabetes_chart_values = [x["high_risk_probability"] for x in diabetes_chart_rows]
        diabetes_chart_predictions = [x["prediction"] for x in diabetes_chart_rows]

        early_diabetes_rows = [
            x for x in timeline
            if str(x.get("disease", "")).strip().lower() == "early diabetes"
        ]
        early_diabetes_chart_rows = early_diabetes_rows[-20:]
        early_diabetes_chart_labels = [x["date"] for x in early_diabetes_chart_rows]
        early_diabetes_chart_values = [x["high_risk_probability"] for x in early_diabetes_chart_rows]
        early_diabetes_chart_predictions = [x["prediction"] for x in early_diabetes_chart_rows]

        # Separate disease analytics for the profile UI.
        heart_count = len(heart_rows)
        heart_high_count = sum(1 for x in heart_rows if "high" in x["prediction"].lower())
        diabetes_count = len(diabetes_rows)
        diabetes_high_count = sum(1 for x in diabetes_rows if "high" in x["prediction"].lower())
        heart_latest = heart_rows[-1] if heart_rows else None
        diabetes_latest = diabetes_rows[-1] if diabetes_rows else None
        heart_recent_rows = list(reversed(heart_rows[-5:]))
        diabetes_recent_rows = list(reversed(diabetes_rows[-5:]))
        early_diabetes_count = len(early_diabetes_rows)
        early_diabetes_high_count = sum(1 for x in early_diabetes_rows if "high" in x["prediction"].lower())
        early_diabetes_latest = early_diabetes_rows[-1] if early_diabetes_rows else None
        early_diabetes_recent_rows = list(reversed(early_diabetes_rows[-5:]))
        early_diabetes_high_probs = [x["high_risk_probability"] for x in early_diabetes_rows]
        avg_early_diabetes_high_probability = round(sum(early_diabetes_high_probs) / len(early_diabetes_high_probs), 2) if early_diabetes_high_probs else None

        # Disease-level counts keep this page ready for future modules.
        disease_summary = {}
        for item in timeline:
            raw_name = str(item.get("disease", "")).strip()
            normalized_name = raw_name.lower().replace("_", " ")

            if normalized_name in ("heart", "heart disease"):
                name = "Heart Disease"
            elif normalized_name == "diabetes":
                name = "Diabetes"
            elif normalized_name == "early diabetes":
                name = "Early Diabetes"
            else:
                name = raw_name
            bucket = disease_summary.setdefault(name, {"total": 0, "high": 0, "latest": None})
            bucket["total"] += 1
            if "high" in item["prediction"].lower():
                bucket["high"] += 1
            bucket["latest"] = item

        return render_template(
            "health_profile.html",
            fullname=fullname,
            email=email,
            total_assessments=total_assessments,
            high_count=high_count,
            low_count=low_count,
            avg_high_probability=avg_high_probability,
            avg_diabetes_high_probability=avg_diabetes_high_probability,
            first=first,
            latest=latest,
            previous=previous,
            latest_change=latest_change,
            comparison=comparison,
            timeline=list(reversed(timeline)),
            disease_summary=disease_summary,
            chart_labels=chart_labels,
            chart_values=chart_values,
            chart_predictions=chart_predictions,
            diabetes_chart_labels=diabetes_chart_labels,
            diabetes_chart_values=diabetes_chart_values,
            diabetes_chart_predictions=diabetes_chart_predictions,
            early_diabetes_chart_labels=early_diabetes_chart_labels,
            early_diabetes_chart_values=early_diabetes_chart_values,
            early_diabetes_chart_predictions=early_diabetes_chart_predictions,
            heart_count=heart_count,
            heart_high_count=heart_high_count,
            heart_rows=heart_rows,
            diabetes_count=diabetes_count,
            diabetes_high_count=diabetes_high_count,
            diabetes_rows=diabetes_rows,
            heart_latest=heart_latest,
            diabetes_latest=diabetes_latest,
            heart_recent_rows=heart_recent_rows,
            diabetes_recent_rows=diabetes_recent_rows,
            early_diabetes_count=early_diabetes_count,
            early_diabetes_high_count=early_diabetes_high_count,
            avg_early_diabetes_high_probability=avg_early_diabetes_high_probability,
            early_diabetes_rows=early_diabetes_rows,
            early_diabetes_latest=early_diabetes_latest,
            early_diabetes_recent_rows=early_diabetes_recent_rows,
            heart_chart_labels=chart_labels,
            heart_chart_values=chart_values,
            heart_chart_predictions=chart_predictions,
        )

    except Exception as e:
        print("HEALTH PROFILE ERROR:", e)
        try:
            mysql.connection.rollback()
        except Exception:
            pass
        flash("Unable to load your health profile.", "danger")
        return redirect(url_for("dashboard"))
    finally:
        if cur:
            cur.close()


# ============================================================
# PREDICTION HISTORY
# ============================================================

@app.route("/history")
def history():

    if not login_required():
        return redirect(url_for("login"))

    try:

        cur = mysql.connection.cursor()

        cur.execute(
            """
            SELECT
                id,
                disease_name,
                prediction,
                confidence,
                high_risk_probability,
                prediction_date
            FROM predictions
            WHERE user_id = %s
            ORDER BY prediction_date DESC
            """,
            (session["user_id"],)
        )

        history_rows = cur.fetchall()

        cur.close()

        # ============================================================
        # Helper: identify disease category
        # ============================================================

        def disease_type(disease_name):

            name = str(disease_name or "").strip().lower()

            if "early" in name and "diabetes" in name:
                return "early_diabetes"

            if "diabetes" in name:
                return "diabetes"

            if "heart" in name:
                return "heart"

            return "other"

        # ============================================================
        # Helper: obtain high-risk probability
        # ============================================================

        def get_high_risk_probability(row):

            prediction = str(row[2] or "").strip().lower()

            confidence = row[3]
            high_risk_probability = row[4]

            # New system:
            # Use explicitly stored high-risk probability.
            if high_risk_probability is not None:

                try:
                    return round(float(high_risk_probability), 2)

                except (TypeError, ValueError):
                    pass

            # Legacy fallback:
            # Older records may not contain high_risk_probability.
            try:

                confidence = float(confidence)

                if confidence <= 1:
                    confidence = confidence * 100

                if "high" in prediction:
                    return round(confidence, 2)

                return round(100 - confidence, 2)

            except (TypeError, ValueError):

                return 0.0

        # ============================================================
        # Separate disease histories
        # ============================================================

        heart_history = []
        diabetes_history = []
        early_diabetes_history = []
        other_history = []

        for row in history_rows:

            category = disease_type(row[1])

            if category == "heart":

                heart_history.append(row)

            elif category == "diabetes":

                diabetes_history.append(row)

            elif category == "early_diabetes":

                early_diabetes_history.append(row)

            else:

                other_history.append(row)

        # ============================================================
        # Counts
        # ============================================================

        total_count = len(history_rows)

        heart_count = len(heart_history)

        diabetes_count = len(diabetes_history)

        early_diabetes_count = len(early_diabetes_history)

        # ============================================================
        # Create trend data
        # ============================================================

        def build_trend(rows):

            chronological = list(reversed(rows))

            labels = []
            values = []
            predictions = []

            for row in chronological:

                if row[5]:

                    labels.append(
                        row[5].strftime("%d %b %Y")
                    )

                else:

                    labels.append("N/A")

                values.append(
                    get_high_risk_probability(row)
                )

                predictions.append(
                    str(row[2] or "")
                )

            return labels, values, predictions

        # Heart trend
        heart_trend_labels, heart_trend_values, heart_trend_predictions = \
            build_trend(heart_history)

        # Diabetes trend
        diabetes_trend_labels, diabetes_trend_values, diabetes_trend_predictions = \
            build_trend(diabetes_history)

        # Early diabetes trend
        early_diabetes_trend_labels, early_diabetes_trend_values, \
            early_diabetes_trend_predictions = build_trend(
                early_diabetes_history
            )

        # ============================================================
        # Latest assessment
        # ============================================================

        latest_prediction_id = (
            history_rows[0][0]
            if history_rows
            else None
        )

        return render_template(
            "history.html",

            # Complete history
            history=history_rows,

            # Disease-specific history
            heart_history=heart_history,
            diabetes_history=diabetes_history,
            early_diabetes_history=early_diabetes_history,
            other_history=other_history,

            # Counts
            total_count=total_count,
            heart_count=heart_count,
            diabetes_count=diabetes_count,
            early_diabetes_count=early_diabetes_count,

            # Latest assessment
            latest_prediction_id=latest_prediction_id,

            # Heart trend
            heart_trend_labels=heart_trend_labels,
            heart_trend_values=heart_trend_values,
            heart_trend_predictions=heart_trend_predictions,

            # Diabetes trend
            diabetes_trend_labels=diabetes_trend_labels,
            diabetes_trend_values=diabetes_trend_values,
            diabetes_trend_predictions=diabetes_trend_predictions,

            # Early Diabetes trend
            early_diabetes_trend_labels=early_diabetes_trend_labels,
            early_diabetes_trend_values=early_diabetes_trend_values,
            early_diabetes_trend_predictions=early_diabetes_trend_predictions
        )

    except Exception as e:

        print(
            "HISTORY ERROR:",
            e
        )

        flash(
            "Unable to load prediction history.",
            "danger"
        )

        return redirect(
            url_for("dashboard")
        )

# ============================================================
@app.route('/user-manual')
def user_manual():
    return render_template('user_manual.html')

# ============================================================
# VIEW PREDICTION HISTORY DETAIL
# ============================================================

@app.route("/prediction_detail/<int:prediction_id>")
def prediction_detail(prediction_id):

    if not login_required():
        return redirect(url_for("login"))

    try:
        cur = mysql.connection.cursor()
        cur.execute(
            """
            SELECT id, disease_name, prediction, confidence, high_risk_probability, input_data, prediction_date
            FROM predictions
            WHERE id = %s AND user_id = %s
            LIMIT 1
            """,
            (prediction_id, session["user_id"])
        )
        row = cur.fetchone()
        cur.close()

        if not row:
            flash("Prediction record not found.", "warning")
            return redirect(url_for("history"))

        try:
            user_data = json.loads(row[5]) if row[5] else {}
        except (TypeError, json.JSONDecodeError):
            user_data = {}

        return render_template(
            "prediction_detail.html",
            prediction_id=row[0],
            disease_name=row[1],
            prediction=row[2],
            confidence=row[3],
            high_risk_probability=row_high_risk_probability(row[2], row[3], row[4]),
            user_data=user_data,
            prediction_date=row[6]
        )

    except Exception as e:
        print("PREDICTION DETAIL ERROR:", e)
        try:
            mysql.connection.rollback()
        except Exception:
            pass
        flash("Unable to load prediction details.", "danger")
        return redirect(url_for("history"))


# ============================================================
# MEDICAL REPORTS
# ============================================================

@app.route("/reports")
def reports():

    if not login_required():

        return redirect(
            url_for("login")
        )


    try:

        cur = mysql.connection.cursor()


        cur.execute(

            """
            SELECT
                id,
                report_name,
                report_type,
                extraction_status,
                extracted_fields,
                missing_fields,
                upload_date
            FROM medical_reports
            WHERE user_id = %s
            ORDER BY upload_date DESC
            """,

            (session["user_id"],)

        )


        reports_data = cur.fetchall()


        cur.close()


        return render_template(

            "reports.html",

            reports=reports_data

        )


    except Exception as e:

        print(
            "REPORTS ERROR:",
            e
        )

        flash(
            "Unable to load medical reports.",
            "danger"
        )

        return redirect(
            url_for("dashboard")
        )


# ============================================================
# VIEW MEDICAL REPORT
# ============================================================

@app.route(
    "/view_report/<int:report_id>"
)
def view_report(report_id):

    if not login_required():

        return redirect(
            url_for("login")
        )


    try:

        cur = mysql.connection.cursor()


        cur.execute(

            """
            SELECT file_path
            FROM medical_reports
            WHERE id = %s
            AND user_id = %s
            """,

            (

                report_id,

                session["user_id"]

            )

        )


        report = cur.fetchone()


        cur.close()


        if not report:

            flash(
                "Medical report not found.",
                "danger"
            )

            return redirect(
                url_for("reports")
            )


        filepath = report[0]


        if (

            not filepath

            or not os.path.exists(filepath)

        ):

            flash(

                "The uploaded medical report "
                "file could not be found.",

                "danger"

            )

            return redirect(
                url_for("reports")
            )


        directory = os.path.dirname(
            filepath
        )

        filename = os.path.basename(
            filepath
        )


        return send_from_directory(

            directory,

            filename

        )


    except Exception as e:

        print(
            "VIEW REPORT ERROR:",
            e
        )

        flash(
            "Unable to view medical report.",
            "danger"
        )

        return redirect(
            url_for("reports")
        )


# ============================================================
# DELETE MEDICAL REPORT
# ============================================================

@app.route(
    "/delete_report/<int:report_id>",
    methods=["POST"]
)
def delete_report(report_id):

    if not login_required():

        return redirect(
            url_for("login")
        )


    try:

        cur = mysql.connection.cursor()


        # -----------------------------------------------------
        # Get file path
        # -----------------------------------------------------

        cur.execute(

            """
            SELECT file_path
            FROM medical_reports
            WHERE id = %s
            AND user_id = %s
            """,

            (

                report_id,

                session["user_id"]

            )

        )


        report = cur.fetchone()


        if not report:

            cur.close()

            flash(
                "Medical report not found.",
                "danger"
            )

            return redirect(
                url_for("reports")
            )


        filepath = report[0]


        # -----------------------------------------------------
        # Delete database record
        # -----------------------------------------------------

        cur.execute(

            """
            DELETE FROM medical_reports
            WHERE id = %s
            AND user_id = %s
            """,

            (

                report_id,

                session["user_id"]

            )

        )


        mysql.connection.commit()

        cur.close()


        # -----------------------------------------------------
        # Delete physical file
        # -----------------------------------------------------

        if (

            filepath

            and os.path.exists(filepath)

        ):

            try:

                os.remove(filepath)

            except Exception as file_error:

                print(
                    "FILE DELETE ERROR:",
                    file_error
                )


        flash(

            "Medical report deleted successfully.",

            "success"

        )


        return redirect(
            url_for("reports")
        )


    except Exception as e:

        mysql.connection.rollback()

        print(
            "DELETE REPORT ERROR:",
            e
        )

        flash(
            "Unable to delete medical report.",
            "danger"
        )

        return redirect(
            url_for("reports")
        )


# ============================================================
# USE SAVED MEDICAL REPORT
# ============================================================

@app.route(
    "/use_report/<int:report_id>"
)
def use_report(report_id):

    if not login_required():

        return redirect(
            url_for("login")
        )


    try:

        cur = mysql.connection.cursor()


        # -----------------------------------------------------
        # Get saved report
        # -----------------------------------------------------

        cur.execute(

            """
            SELECT file_path, report_type
            FROM medical_reports
            WHERE id = %s
            AND user_id = %s
            """,

            (

                report_id,

                session["user_id"]

            )

        )


        report = cur.fetchone()


        cur.close()


        if not report:

            flash(
                "Medical report not found.",
                "danger"
            )

            return redirect(
                url_for("reports")
            )


        filepath = report[0]
        saved_report_type = str(report[1] or "").lower()

        # Diabetes reports use the diabetes extraction form. This branch is
        # isolated so the existing Heart Disease reuse flow remains unchanged.
        if "diabetes" in saved_report_type:
            text = extract_text(filepath)
            extracted_data = parse_diabetes_report(text) if text else {f: None for f in DIABETES_FIELDS}
            extracted_count = sum(v is not None for v in extracted_data.values())
            missing_count = len(DIABETES_FIELDS) - extracted_count
            return render_template("diabetes_prediction.html", extracted_data=extracted_data, report_type=saved_report_type, extracted_count=extracted_count, missing_count=missing_count, report_id=report_id)

        # -----------------------------------------------------
        # Check file
        # -----------------------------------------------------

        if (

            not filepath

            or not os.path.exists(filepath)

        ):

            flash(

                "The saved medical report file "
                "could not be found.",

                "danger"

            )

            return redirect(
                url_for("reports")
            )


        # -----------------------------------------------------
        # Extract PDF text
        # -----------------------------------------------------

        text = extract_text(
            filepath
        )


        if (

            not text

            or len(text.strip()) < 20

        ):

            flash(

                "Unable to read meaningful text "
                "from this report.",

                "danger"

            )

            return redirect(
                url_for("reports")
            )


        # -----------------------------------------------------
        # Detect report type
        # -----------------------------------------------------

        report_type = detect_report_type(
            text
        )


        # -----------------------------------------------------
        # Parse report
        # -----------------------------------------------------

        extracted_data = parse_report(
            text
        )


        # -----------------------------------------------------
        # Normalize
        # -----------------------------------------------------

        extracted_data = normalize_extracted_data(

            extracted_data

        )


        # -----------------------------------------------------
        # Ensure all fields exist
        # -----------------------------------------------------

        for field in EXPECTED_FIELDS:

            if field not in extracted_data:

                extracted_data[field] = None


        # -----------------------------------------------------
        # Count extracted fields
        # -----------------------------------------------------

        extracted_count = sum(

            1

            for field in EXPECTED_FIELDS

            if extracted_data.get(field) is not None

        )


        missing_count = sum(

            1

            for field in EXPECTED_FIELDS

            if extracted_data.get(field) is None

        )


        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------

        validation = validate_extracted_data(

            extracted_data

        )


        # -----------------------------------------------------
        # Compatibility
        # -----------------------------------------------------

        compatibility = (
            check_heart_model_compatibility(
                extracted_data
            )
        )


        # -----------------------------------------------------
        # Debug
        # -----------------------------------------------------

        print("=" * 70)

        print(
            "REUSING SAVED MEDICAL REPORT"
        )

        print("=" * 70)

        print(
            "Report ID:",
            report_id
        )

        print(
            "File:",
            filepath
        )

        print(
            "Report Type:",
            report_type
        )

        print(
            "Extracted Data:",
            extracted_data
        )

        print(
            "Extracted Fields:",
            extracted_count
        )

        print(
            "Missing Fields:",
            missing_count
        )

        print(
            "Validation:",
            validation
        )

        print(
            "Compatibility:",
            compatibility
        )

        print("=" * 70)


        # -----------------------------------------------------
        # Show prediction form
        # -----------------------------------------------------

        return render_template(

            "prediction.html",

            extracted_data=extracted_data,

            validation=validation,

            compatibility=compatibility,

            report_type=report_type,

            extracted_count=extracted_count,

            missing_count=missing_count,

            report_id=report_id

        )


    except Exception as e:

        print(
            "USE REPORT ERROR:",
            e
        )

        flash(

            "Unable to reuse the medical report.",

            "danger"

        )

        return redirect(
            url_for("reports")
        )


# ============================================================
# UPLOAD MEDICAL REPORT
# ============================================================

@app.route(
    "/upload_report",
    methods=["POST"]
)
def upload_report():

    if not login_required():

        return redirect(
            url_for("login")
        )


    if "report" not in request.files:

        flash(
            "No medical report selected.",
            "danger"
        )

        return redirect(
            url_for("heart_prediction")
        )


    file = request.files["report"]


    if file.filename == "":

        flash(
            "Please select a medical report.",
            "warning"
        )

        return redirect(
            url_for("heart_prediction")
        )


    if not file.filename.lower().endswith(".pdf"):

        flash(
            "Only PDF reports are supported.",
            "danger"
        )

        return redirect(
            url_for("heart_prediction")
        )


    try:

        # =====================================================
        # SAVE PDF
        # =====================================================

        original_filename = secure_filename(
            file.filename
        )


        # Add timestamp to avoid filename conflicts

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )


        filename = (
            f"{timestamp}_{original_filename}"
        )


        filepath = os.path.join(

            app.config["UPLOAD_FOLDER"],

            filename

        )


        file.save(
            filepath
        )


        # =====================================================
        # EXTRACT PDF TEXT
        # =====================================================

        text = extract_text(
            filepath
        )


        print("=" * 70)
        print("EXTRACTED PDF TEXT")
        print("=" * 70)
        print(text)
        print("=" * 70)


        if (

            not text

            or len(text.strip()) < 20

        ):

            flash(

                "Unable to read meaningful text "
                "from the PDF.",

                "danger"

            )

            # Remove invalid file

            if os.path.exists(filepath):

                os.remove(filepath)


            return redirect(
                url_for("heart_prediction")
            )


        # =====================================================
        # DETECT REPORT TYPE
        # =====================================================

        report_type = detect_report_type(
            text
        )


        print(
            "Detected Report Type:",
            report_type
        )


        # =====================================================
        # PARSE REPORT
        # =====================================================

        extracted_data = parse_report(
            text
        )


        print(
            "Raw Parsed Data:",
            extracted_data
        )


        # =====================================================
        # NORMALIZE
        # =====================================================

        extracted_data = normalize_extracted_data(

            extracted_data

        )


        # =====================================================
        # ENSURE ALL MODEL FIELDS
        # =====================================================

        for field in EXPECTED_FIELDS:

            if field not in extracted_data:

                extracted_data[field] = None


        # =====================================================
        # VALIDATION
        # =====================================================

        validation = validate_extracted_data(

            extracted_data

        )


        # =====================================================
        # COUNT FIELDS
        # =====================================================

        extracted_count = sum(

            1

            for field in EXPECTED_FIELDS

            if extracted_data.get(field) is not None

        )


        missing_count = sum(

            1

            for field in EXPECTED_FIELDS

            if extracted_data.get(field) is None

        )


        # =====================================================
        # SAVE REPORT TO DATABASE
        # =====================================================

        cur = mysql.connection.cursor()


        extraction_status = (

            "Complete"

            if missing_count == 0

            else "Incomplete"

        )


        cur.execute(

            """
            INSERT INTO medical_reports
            (
                user_id,
                report_name,
                report_type,
                file_path,
                extraction_status,
                extracted_fields,
                missing_fields,
                extracted_data
            )
            VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s)
            """,

            (

                session["user_id"],

                original_filename,

                report_type,

                filepath,

                extraction_status,

                extracted_count,

                missing_count,

                json.dumps(extracted_data)

            )

        )


        mysql.connection.commit()


        report_id = cur.lastrowid


        cur.close()

        
        # ============================================================
        # CREATE REPORT NOTIFICATION
        # ============================================================

        if missing_count > 0:

            create_notification(

                session["user_id"],

                "Medical Report Analyzed",

                f"Your medical report was uploaded successfully. "
                f"{extracted_count} clinical parameter(s) were extracted. "
                f"{missing_count} parameter(s) still need to be entered manually.",

                "warning",
                reference_type="report", reference_id=report_id, priority="attention"

            )

        else:

            create_notification(

                session["user_id"],

                "Medical Report Ready",

                "Your medical report was uploaded successfully and "
                "all required clinical parameters were extracted. "
                "Please verify the values before prediction.",

                "success",
                reference_type="report", reference_id=report_id, priority="normal"

            )


        # =====================================================
        # MODEL COMPATIBILITY
        # =====================================================

        compatibility = (
            check_heart_model_compatibility(
                extracted_data
            )
        )


        # =====================================================
        # DEBUG
        # =====================================================

        print("=" * 70)

        print("REPORT ANALYSIS")

        print("=" * 70)

        print(
            "Report Type:",
            report_type
        )


        print(
            "\nExtracted Data:"
        )


        for field in EXPECTED_FIELDS:

            print(

                f"{field}: "
                f"{extracted_data.get(field)}"

            )


        print(
            "\nExtracted Fields:",
            extracted_count
        )

        print(
            "Missing Fields:",
            missing_count
        )


        print(
            "\nValidation:",
            validation
        )


        print(
            "\nHeart Model Compatibility:",
            compatibility
        )


        print("=" * 70)


        # =====================================================
        # USER MESSAGE
        # =====================================================

        if extracted_count == 0:

            flash(

                "The report text was readable, but no "
                "supported clinical parameters could be "
                "extracted. Please enter the values manually.",

                "warning"

            )


        elif missing_count > 0:

            flash(

                f"{extracted_count} clinical parameter(s) "
                f"extracted successfully. Please enter "
                f"the remaining {missing_count} parameter(s) "
                f"manually.",

                "warning"

            )


        else:

            flash(

                "All required clinical parameters were "
                "extracted. Please verify the values "
                "before prediction.",

                "success"

            )


        # =====================================================
        # DISPLAY PREDICTION FORM
        # =====================================================

        return render_template(

            "prediction.html",

            extracted_data=extracted_data,

            validation=validation,

            compatibility=compatibility,

            report_type=report_type,

            extracted_count=extracted_count,

            missing_count=missing_count,

            report_id=report_id

        )


    except Exception as e:

        mysql.connection.rollback()


        print("=" * 70)

        print(
            "REPORT PROCESSING ERROR"
        )

        print("=" * 70)

        print(e)

        print("=" * 70)


        flash(

            f"Unable to process medical report: {e}",

            "danger"

        )


        return redirect(
            url_for("heart_prediction")
        )


# ============================================================
# NOTIFICATIONS
# ============================================================

# ============================================================
# NOTIFICATIONS
# ============================================================

@app.route("/notifications")
def notifications():

    if not login_required():

        return redirect(
            url_for("login")
        )

    user_id = session["user_id"]

    try:

        cur = mysql.connection.cursor()

        cur.execute(
            """
            SELECT
                id, title, message, notification_type, is_read, created_at,
                reference_type, reference_id, priority
            FROM notifications
            WHERE user_id = %s
            ORDER BY created_at DESC
            """,
            (user_id,)
        )

        notifications_data = cur.fetchall()

        cur.close()

        cur = mysql.connection.cursor()
        cur.execute("SELECT COUNT(*) FROM notifications WHERE user_id = %s", (user_id,))
        total_notifications = int(cur.fetchone()[0] or 0)
        cur.execute("SELECT COUNT(*) FROM notifications WHERE user_id = %s AND is_read = 0", (user_id,))
        unread_count = int(cur.fetchone()[0] or 0)
        cur.execute("""
            SELECT COUNT(*) FROM notifications
            WHERE user_id = %s
              AND (title LIKE 'Risk Probability %%'
                   OR title = 'Prediction Classification Changed'
                   OR title = 'Assessment Values Changed')
        """, (user_id,))
        smart_alert_count = int(cur.fetchone()[0] or 0)
        cur.close()

        return render_template(
            "notifications.html",
            notifications=notifications_data,
            total_notifications=total_notifications,
            unread_count=unread_count,
            smart_alert_count=smart_alert_count
        )

    except Exception as e:

        print(
            "NOTIFICATIONS ERROR:",
            e
        )

        flash(
            "Unable to load notifications.",
            "danger"
        )

        return redirect(
            url_for("dashboard")
        )


# ============================================================
# MARK SINGLE NOTIFICATION AS READ
# ============================================================

@app.route(
    "/notifications/read/<int:notification_id>",
    methods=["POST"]
)
def mark_notification_read(notification_id):

    if not login_required():

        return redirect(
            url_for("login")
        )

    try:

        cur = mysql.connection.cursor()

        cur.execute(
            """
            UPDATE notifications
            SET is_read = 1
            WHERE id = %s
            AND user_id = %s
            """,
            (
                notification_id,
                session["user_id"]
            )
        )

        mysql.connection.commit()

        cur.close()

    except Exception as e:

        mysql.connection.rollback()

        print(
            "MARK NOTIFICATION READ ERROR:",
            e
        )

    return redirect(
        url_for("notifications")
    )


@app.route("/notifications/mark-all-read")
def mark_all_notifications_read():

    if not login_required():
        return redirect(url_for("login"))

    user_id = session["user_id"]

    cur = mysql.connection.cursor()

    cur.execute("""
        UPDATE notifications
        SET is_read = 1
        WHERE user_id = %s
    """, (user_id,))

    mysql.connection.commit()

    cur.close()

    return redirect(url_for("notifications"))
# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()


    flash(
        "Logged out successfully!",
        "success"
    )


    return redirect(
        url_for("home")
    )


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return render_template(
        "404.html"
    ), 404


@app.errorhandler(413)
def request_entity_too_large(error):

    flash(
        "The uploaded file is too large. "
        "Maximum allowed size is 10 MB.",
        "danger"
    )

    return redirect(
        url_for("heart_prediction")
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )