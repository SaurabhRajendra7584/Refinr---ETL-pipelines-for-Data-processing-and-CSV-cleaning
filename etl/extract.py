# etl/extract.py

import pandas as pd
from utils.logger import logger

def read_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully read CSV: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        return None

def read_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Successfully read Excel: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading Excel: {e}")
        return None

def read_json(file_path):
    try:
        df = pd.read_json(file_path)
        logger.info(f"Successfully read JSON: {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error reading JSON: {e}")
        return None
