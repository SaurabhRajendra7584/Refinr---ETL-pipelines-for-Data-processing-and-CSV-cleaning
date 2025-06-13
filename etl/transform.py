# etl/transform.py

import pandas as pd
import numpy as np
from utils.logger import logger
from sklearn.preprocessing import StandardScaler

# Handle missing values
def handle_missing_values(df, method='drop', fill_value=0):
    try:
        if method == 'drop':
            df_clean = df.dropna()
        elif method == 'fill':
            df_clean = df.fillna(fill_value)
        else:
            logger.warning("Unknown method passed to handle_missing_values")
            df_clean = df
        logger.info("Missing values handled")
        return df_clean
    except Exception as e:
        logger.error(f"Error in handle_missing_values: {e}")
        return df

# Remove outliers using IQR method
def remove_outliers_iqr(df, columns):
    try:
        for col in columns:
            if np.issubdtype(df[col].dtype, np.number):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower) & (df[col] <= upper)]
        logger.info("Outliers removed using IQR")
        return df
    except Exception as e:
        logger.error(f"Error removing outliers: {e}")
        return df

# Encode categorical variables using one-hot encoding
def encode_categorical(df, columns):
    try:
        df_encoded = pd.get_dummies(df, columns=columns, drop_first=True)
        logger.info("Categorical encoding done")
        return df_encoded
    except Exception as e:
        logger.error(f"Error encoding categorical variables: {e}")
        return df

# Scale numeric features
def scale_numeric(df, columns):
    try:
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        logger.info("Numeric scaling done")
        return df
    except Exception as e:
        logger.error(f"Error scaling numeric features: {e}")
        return df

# Convert data types
def convert_dtypes(df, conversions: dict):
    try:
        for col, dtype in conversions.items():
            df[col] = df[col].astype(dtype)
        logger.info("Data type conversion successful")
        return df
    except Exception as e:
        logger.error(f"Error converting data types: {e}")
        return df
