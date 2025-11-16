\"\"\"Preprocessing utilities.
- Build a sklearn Pipeline that handles any categorical features (one-hot) and passes numeric features through.
- Returns a pipeline ready to fit a GBM and a list of output feature names.
\"\"\"
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def _get_feature_lists(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return numeric_cols, categorical_cols

def build_pipeline_and_features(X: pd.DataFrame, random_state=42):
    numeric_cols, categorical_cols = _get_feature_lists(X)

    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    feature_names = numeric_cols + categorical_cols

    return pipeline, feature_names
