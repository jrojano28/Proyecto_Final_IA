# utils/preprocessing.py
"""Utility functions for data preprocessing.

- Impute missing numeric values with median, categorical with mode.
- Encode categorical columns using sklearn's LabelEncoder.

Both functions return the transformed dataframe and any auxiliary objects needed for later inference.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any

def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in a DataFrame.

    Numeric columns: median.
    Categorical/object columns: mode (most frequent).
    """
    df_filled = df.copy()
    for col in df_filled.columns:
        if pd.api.types.is_numeric_dtype(df_filled[col]):
            median = df_filled[col].median()
            df_filled[col].fillna(median, inplace=True)
        else:
            mode = df_filled[col].mode()
            if not mode.empty:
                df_filled[col].fillna(mode.iloc[0], inplace=True)
            else:
                df_filled[col].fillna("", inplace=True)
    return df_filled

def encode_categorical(
    df: pd.DataFrame, encoders: Dict[str, LabelEncoder] = None
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical columns with LabelEncoder.

    If ``encoders`` is provided, reuse existing encoders (useful for inference).
    Returns transformed DataFrame and a dictionary of encoders.
    """
    df_enc = df.copy()
    if encoders is None:
        encoders = {}
    for col in df_enc.columns:
        if pd.api.types.is_object_dtype(df_enc[col]):
            le = encoders.get(col)
            if le is None:
                le = LabelEncoder()
                df_enc[col] = le.fit_transform(df_enc[col].astype(str))
                encoders[col] = le
            else:
                df_enc[col] = le.transform(df_enc[col].astype(str))
    return df_enc, encoders
