\"\"\"Model training, evaluation and interpretation utilities for Gradient Boosting.
\"\"\"
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from joblib import dump

def _plot_confusion(cm, labels, fname):
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(fname)
    plt.close()

def _plot_roc(y_test, y_proba, fname):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.savefig(fname)
    plt.close()

def _expand_feature_names(preprocessor, feature_names):
    numeric_cols = preprocessor.transformers_[0][2]
    cat_transformer = preprocessor.transformers_[1][1]
    cat_cols = preprocessor.transformers_[1][2]

    expanded = list(numeric_cols)
    if len(cat_cols) > 0:
        ohe = cat_transformer.named_steps['ohe']
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        expanded += ohe_names
    return expanded

def train_and_evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, feature_names, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state)

    gbm = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=random_state,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )

    full_pipeline = Pipeline(steps=[('preprocessor', pipeline.named_steps['preprocessor']),
                                    ('gbm', gbm)])

    full_pipeline.fit(X_train, y_train)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(gbm, pipeline.fit_transform(X_train), y_train, cv=skf, scoring='roc_auc')

    y_pred = full_pipeline.predict(X_test)
    y_proba = full_pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_proba)),
        'cv_roc_auc_mean': float(np.mean(cv_scores)),
        'cv_roc_auc_std': float(np.std(cv_scores)),
    }

    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion(cm, labels=['negative', 'positive'], fname='figures/confusion_matrix.png')
    _plot_roc(y_test, y_proba, fname='figures/roc_curve.png')

    preprocessor = full_pipeline.named_steps['preprocessor']
    expanded_names = _expand_feature_names(preprocessor, feature_names)

    importances = full_pipeline.named_steps['gbm'].feature_importances_
    fi_df = pd.DataFrame({'feature': expanded_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)

    dump(full_pipeline, 'results/gbm_pipeline.joblib')

    return metrics, fi_df
