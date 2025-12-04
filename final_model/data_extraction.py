import sqlite3
import pandas as pd
import os

DB_PATH = 'MIMIC4-hosp-icu.db'
OUTPUT_DIR = 'final_model'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_cancer_cohort():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    
    # Define Cancer ICD Codes Query
    # ICD-9: 140-239 (140% to 239%)
    # ICD-10: C00-D49 (C% to D4%) - Note: D49 is included in D4%
    
    print("Querying diagnoses_icd for cancer patients...")
    query = """
    SELECT DISTINCT subject_id, hadm_id
    FROM diagnoses_icd
    WHERE 
        (icd_version = 9 AND (
            icd_code LIKE '14%' OR icd_code LIKE '15%' OR icd_code LIKE '16%' OR 
            icd_code LIKE '17%' OR icd_code LIKE '18%' OR icd_code LIKE '19%' OR 
            icd_code LIKE '20%' OR icd_code LIKE '21%' OR icd_code LIKE '22%' OR 
            icd_code LIKE '23%'
        ))
        OR
        (icd_version = 10 AND (
            icd_code LIKE 'C%' OR icd_code LIKE 'D0%' OR icd_code LIKE 'D1%' OR 
            icd_code LIKE 'D2%' OR icd_code LIKE 'D3%' OR icd_code LIKE 'D4%'
        ))
    """
    
    cohort_df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Found {len(cohort_df)} unique admissions with cancer diagnoses.")
    
    # Save Cohort
    cohort_path = os.path.join(OUTPUT_DIR, 'cohort.csv')
    cohort_df.to_csv(cohort_path, index=False)
    print(f"Saved cohort to {cohort_path}")
    
    return cohort_df

if __name__ == "__main__":
    extract_cancer_cohort()
