import pandas as pd
import numpy as np
import os

OUTPUT_DIR = 'final_model'
COHORT_PATH = os.path.join(OUTPUT_DIR, 'cohort.csv')
STATIC_SOURCE = 'cohort_static (1).csv' # Using the larger/rawer static file if possible, or processed
DYNAMIC_SOURCE = 'FINAL_CANCER_DATASET_LANDMARK.csv'

def process_data():
    print("Loading Cohort...")
    if not os.path.exists(COHORT_PATH):
        print(f"Cohort file not found at {COHORT_PATH}")
        return
    
    cohort_df = pd.read_csv(COHORT_PATH)
    cohort_ids = set(cohort_df['hadm_id'].unique())
    print(f"Cohort Size: {len(cohort_ids)}")
    
    # 1. Process Static Data
    print("Processing Static Data...")
    # We need demographics. cohort_static (1).csv likely has them.
    # Let's inspect columns of cohort_static (1).csv first to be sure, but assuming it has standard fields.
    # If not, we might need to use cohort_static_processed.csv but drop the cancer_type_NONE column if we don't want it.
    
    # For now, let's try to use cohort_static_processed.csv as it's already cleaned, 
    # but we will filter it by our NEW cohort IDs.
    # If the new cohort has IDs NOT in processed, we might lose them. 
    # Ideally we should go back to raw, but for now let's use the processed one as a base 
    # and see how many we match.
    
    static_df = pd.read_csv('no_text_model/cohort_static_processed.csv')
    
    # Filter by new cohort
    static_df = static_df[static_df['hadm_id'].isin(cohort_ids)]
    
    # Drop cancer_type_NONE if it exists, as we are defining cancer by ICD now
    if 'cancer_type_NONE' in static_df.columns:
        static_df = static_df.drop(columns=['cancer_type_NONE'])
        
    print(f"Static Data Size after filtering: {len(static_df)}")
    static_df.to_csv(os.path.join(OUTPUT_DIR, 'static_data.csv'), index=False)
    
    # 2. Process Dynamic Data
    print("Processing Dynamic Data...")
    # Reading large CSV in chunks if necessary, or just read if memory allows (500MB is okay)
    dynamic_df = pd.read_csv(DYNAMIC_SOURCE)
    
    # Filter
    dynamic_df = dynamic_df[dynamic_df['hadm_id'].isin(cohort_ids)]
    print(f"Dynamic Data Size after filtering: {len(dynamic_df)}")
    
    dynamic_df.to_csv(os.path.join(OUTPUT_DIR, 'dynamic_data.csv'), index=False)
    print("Data Processing Complete.")

if __name__ == "__main__":
    process_data()
