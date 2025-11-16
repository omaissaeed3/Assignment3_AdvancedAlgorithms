\"\"\"Entry point: runs EDA, preprocessing, GBM training and evaluation.\"\"\"
import argparse
import os
import json
import pandas as pd
from preprocessing import build_pipeline_and_features
from model_gbm import train_and_evaluate

def ensure_dirs():
    os.makedirs('figures', exist_ok=True)
    os.makedirs('results', exist_ok=True)

def load_data(path):
    df = pd.read_csv(path)
    return df

def main(data_path, random_state):
    ensure_dirs()
    df = load_data(data_path)

    print('Rows, columns:', df.shape)
    # Try common target column names used in Kaggle dataset
    if 'classification' in df.columns:
        target_col = 'classification'
    elif 'diagnosis' in df.columns:
        target_col = 'diagnosis'
    elif 'target' in df.columns:
        target_col = 'target'
    else:
        # fallback: assume last column is target
        target_col = df.columns[-1]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    pipeline, feature_names = build_pipeline_and_features(X, random_state=random_state)

    metrics, fi_df = train_and_evaluate(pipeline, X, y, feature_names,
                                        random_state=random_state)

    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    fi_df.to_csv('results/feature_importances.csv', index=False)

    print('Done. Metrics and figures saved to results/ and figures/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    main(args.data_path, args.random_state)
