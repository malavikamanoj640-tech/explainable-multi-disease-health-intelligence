@echo off
cd /d "%~dp0"
echo ============================================================
echo AI HEALTH RISK ASSESSMENT - NEW DATABASE SETUP
echo ============================================================
python database_setup.py
if errorlevel 1 (
  echo.
  echo Database setup failed. Check that XAMPP MySQL is running on port 3306.
  pause
  exit /b 1
)
echo.
echo New database setup completed successfully.
pause
