from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_feature_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str]
) -> ColumnTransformer:
    """
    Build a ColumnTransformer which applies different transformations:
      - standardises numerical columns
      - impute then one-hot-encodes categorical columns
    """
    # pipeline for numerical data
    num_pipeline = [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]

    # pipeline for categorical data
    cat_pipeline = [
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]

    return ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numeric_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'  # dropping other columns
    )

def split_X_y(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts X (features) and y (target) from Dataframe.
    """
    return df.drop(columns=target_col), df[target_col]
