import pandas as pd
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
import os

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

# =====================================
# SQL Query (MATCHES YOUR TABLE 100%)
# =====================================
QUERY = """
SELECT
    id,
    image_name,
    CASE 
        WHEN cancer_present = 1 THEN 'Cancer'
        ELSE 'Normal'
    END AS cancer_status,
    presence_conf,
    cancer_type,
    type_conf,
    created_at
FROM predictions
ORDER BY created_at;
"""

# =====================================
# Export Function
# =====================================
def export_to_excel():
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql(QUERY, conn)
    conn.close()

    file_name = "cancer_predictions_report.xlsx"
    df.to_excel(file_name, index=False)

    print("✅ Excel report updated successfully")
    print(f"📄 File: {file_name}")
    print(f"🕒 Updated at: {datetime.now()}")

# =====================================
# Entry Point
# =====================================
if __name__ == "__main__":
    export_to_excel()