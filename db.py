import os
import mysql.connector
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ======================================================
# MySQL Configuration
# ======================================================
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            image_name VARCHAR(255),
            cancer_present BOOLEAN,
            presence_conf FLOAT,
            cancer_type VARCHAR(100),
            type_conf FLOAT,
            created_at DATETIME
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()


def save_prediction(
    image_name,
    cancer_present,
    presence_conf,
    cancer_type=None,
    type_conf=None
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            image_name,
            cancer_present,
            presence_conf,
            cancer_type,
            type_conf,
            created_at
        ) VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        image_name,
        cancer_present,
        presence_conf,
        cancer_type,
        type_conf,
        datetime.now()
    ))

    conn.commit()
    cursor.close()
    conn.close()


# Initialize DB on import
init_db()