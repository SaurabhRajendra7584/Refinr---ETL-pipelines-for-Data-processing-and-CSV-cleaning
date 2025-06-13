# etl/load.py

import pandas as pd
from utils.logger import logger
from sqlalchemy import create_engine

# Save to CSV
def save_to_csv(df, file_path):
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Saved cleaned data to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving CSV: {e}")

# Optional: Load to SQL database
def save_to_database(df, db_uri, table_name):
    try:
        engine = create_engine(db_uri)
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logger.info(f"Saved cleaned data to database: {table_name}")
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
