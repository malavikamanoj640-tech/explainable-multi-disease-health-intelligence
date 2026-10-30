import smtplib
import config

from email.message import EmailMessage


print("Testing SMTP...")
print("Server:", config.MAIL_SERVER)
print("Port:", config.MAIL_PORT)
print("Username:", config.MAIL_USERNAME)


message = EmailMessage()

message["Subject"] = "AI Health System - SMTP Test"

message["From"] = config.MAIL_FROM

message["To"] = config.MAIL_USERNAME

message.set_content(
    """
This is a test email from the
AI Health Risk Assessment System.

If you received this email,
SMTP configuration is working correctly.
"""
)


try:

    with smtplib.SMTP(
        config.MAIL_SERVER,
        config.MAIL_PORT,
        timeout=20
    ) as smtp:

        smtp.ehlo()

        smtp.starttls()

        smtp.ehlo()

        smtp.login(
            config.MAIL_USERNAME,
            config.MAIL_PASSWORD
        )

        smtp.send_message(message)


    print("=" * 60)
    print("EMAIL SENT SUCCESSFULLY")
    print("=" * 60)


except Exception as e:

    print("=" * 60)
    print("EMAIL FAILED")
    print("ERROR TYPE:", type(e).__name__)
    print("ERROR:", str(e))
    print("=" * 60)