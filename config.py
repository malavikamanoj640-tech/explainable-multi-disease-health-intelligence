# ============================================================
# AI HEALTH RISK ASSESSMENT - NEW DATABASE CONFIGURATION
# ============================================================
# This project intentionally uses a NEW database.
# The existing heart_prediction_db is NOT used or modified.

MYSQL_HOST = "127.0.0.1"
MYSQL_PORT = 3306
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "heart_prediction_new_db"
MYSQL_CONNECT_TIMEOUT = 5

SECRET_KEY = "heart_prediction_new_secret"

import os
# ============================================================
# EMAIL / PASSWORD RESET
# ============================================================

MAIL_SERVER = "smtp.gmail.com"
MAIL_PORT = 587
MAIL_USE_TLS = True

MAIL_USERNAME = "your_email"

MAIL_PASSWORD = "your_created_password"

MAIL_FROM = MAIL_USERNAME

