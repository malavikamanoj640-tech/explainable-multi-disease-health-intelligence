"""Create and verify the NEW application database without touching the old one."""
import time
import MySQLdb
import config

DB_NAME = config.MYSQL_DB

TABLES = {
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            fullname VARCHAR(100) NOT NULL,
            email VARCHAR(150) NOT NULL UNIQUE,
            password VARCHAR(255) NOT NULL
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    "password_reset_otps": """
        CREATE TABLE IF NOT EXISTS password_reset_otps (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            otp_hash VARCHAR(255) NOT NULL,
            expires_at DATETIME NOT NULL,
            attempts INT NOT NULL DEFAULT 0,
            used TINYINT(1) NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_reset_user (user_id),
            INDEX idx_reset_expiry (expires_at),
            CONSTRAINT fk_reset_user
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    "prediction_audit": """
        CREATE TABLE IF NOT EXISTS prediction_audit (
            id INT AUTO_INCREMENT PRIMARY KEY,
            prediction_id INT NOT NULL,
            user_id INT NOT NULL,
            disease_name VARCHAR(50) NOT NULL,
            model_name VARCHAR(150) NULL,
            model_version VARCHAR(80) NULL,
            model_file VARCHAR(255) NULL,
            dataset_reference VARCHAR(255) NULL,
            decision_threshold FLOAT DEFAULT 0.50,
            high_risk_probability FLOAT NULL,
            predicted_label VARCHAR(100) NULL,
            input_snapshot LONGTEXT NULL,
            audit_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_prediction_audit_prediction (prediction_id),
            INDEX idx_prediction_audit_user (user_id),
            INDEX idx_prediction_audit_disease (disease_name)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            disease_name VARCHAR(50),
            prediction VARCHAR(100),
            confidence FLOAT,
            high_risk_probability FLOAT NULL,
            input_data LONGTEXT NULL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_predictions_user (user_id),
            INDEX idx_predictions_date (prediction_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    "notifications": """
        CREATE TABLE IF NOT EXISTS notifications (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            title VARCHAR(255) NOT NULL,
            message TEXT NOT NULL,
            notification_type VARCHAR(50) DEFAULT 'info',
            is_read TINYINT(1) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reference_type VARCHAR(50) NULL,
            reference_id INT NULL,
            priority VARCHAR(20) DEFAULT 'normal',
            INDEX idx_notifications_user (user_id),
            INDEX idx_notifications_read (is_read)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
    "medical_reports": """
        CREATE TABLE IF NOT EXISTS medical_reports (
            id INT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NOT NULL,
            report_name VARCHAR(255) NOT NULL,
            report_type VARCHAR(100),
            file_path VARCHAR(500),
            extraction_status VARCHAR(50),
            extracted_fields INT DEFAULT 0,
            missing_fields INT DEFAULT 0,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            extracted_data LONGTEXT NULL,
            INDEX idx_reports_user (user_id),
            INDEX idx_reports_date (upload_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """,
}


def _server_connection():
    return MySQLdb.connect(
        host=config.MYSQL_HOST,
        port=int(getattr(config, "MYSQL_PORT", 3306)),
        user=config.MYSQL_USER,
        passwd=config.MYSQL_PASSWORD,
        connect_timeout=int(getattr(config, "MYSQL_CONNECT_TIMEOUT", 5)),
        charset="utf8mb4",
    )


def ensure_database_and_tables(retries=3, delay=1.0):
    """Create only the configured NEW database and its application tables."""
    last_error = None
    for attempt in range(1, retries + 1):
        server = None
        db = None
        try:
            server = _server_connection()
            cur = server.cursor()
            safe_db = DB_NAME.replace("`", "``")
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{safe_db}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            cur.close()
            server.commit()
            server.close()
            server = None

            db = MySQLdb.connect(
                host=config.MYSQL_HOST,
                port=int(getattr(config, "MYSQL_PORT", 3306)),
                user=config.MYSQL_USER,
                passwd=config.MYSQL_PASSWORD,
                db=DB_NAME,
                connect_timeout=int(getattr(config, "MYSQL_CONNECT_TIMEOUT", 5)),
                charset="utf8mb4",
            )
            cur = db.cursor()
            for sql in TABLES.values():
                cur.execute(sql)
            # Backward-compatible upgrade for a database created by an earlier build.
            # MariaDB/MySQL installations may differ in support for IF NOT EXISTS on ADD COLUMN,
            # so ignore only the duplicate-column condition.
            try:
                cur.execute("ALTER TABLE medical_reports ADD COLUMN extracted_data LONGTEXT NULL")
            except Exception as alter_error:
                if "Duplicate column" not in str(alter_error) and "1060" not in str(alter_error):
                    raise
            # Backward-compatible prediction upgrade.
            try:
                cur.execute("ALTER TABLE predictions ADD COLUMN high_risk_probability FLOAT NULL")
            except Exception as alter_error:
                if "Duplicate column" not in str(alter_error) and "1060" not in str(alter_error):
                    raise
            # Backward-compatible notification upgrades.
            notification_columns = [
                ("reference_type", "VARCHAR(50) NULL"),
                ("reference_id", "INT NULL"),
                ("priority", "VARCHAR(20) DEFAULT 'normal'"),
            ]
            for column_name, column_definition in notification_columns:
                try:
                    cur.execute(
                        f"ALTER TABLE notifications ADD COLUMN {column_name} {column_definition}"
                    )
                except Exception as alter_error:
                    if "Duplicate column" not in str(alter_error) and "1060" not in str(alter_error):
                        raise
            db.commit()
            cur.close()
            db.close()

            print("=" * 70)
            print("NEW DATABASE READY")
            print("Database:", DB_NAME)
            print("Existing heart_prediction_db was NOT modified.")
            print("Tables verified:", ", ".join(TABLES.keys()))
            print("=" * 70)
            return True
        except Exception as exc:
            last_error = exc
            print(f"DATABASE SETUP ATTEMPT {attempt}/{retries} FAILED: {exc}")
            for conn in (server, db):
                try:
                    if conn:
                        conn.close()
                except Exception:
                    pass
            if attempt < retries:
                time.sleep(delay)

    raise RuntimeError(
        f"Could not create/connect to {DB_NAME}. "
        f"MySQL must be running on {config.MYSQL_HOST}:{getattr(config, 'MYSQL_PORT', 3306)}. "
        f"Last error: {last_error}"
    )


if __name__ == "__main__":
    ensure_database_and_tables()
