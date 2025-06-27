import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from feature_selection import select_features


def prepare_features_and_targets(train_df, label, labels_df=None, feature_columns=None):
    feature_columns = select_features(train_df, labels_df, label) if labels_df is not None else feature_columns
    X = train_df[feature_columns].copy()
    for col in feature_columns:
        if col.startswith('o_'):
            non_null_mask = X[col].notna()
            X.loc[non_null_mask, col] = X.loc[non_null_mask, col].astype(int)

    for col in feature_columns:
        if col.startswith('c_') or col.startswith('o_'):
            X[col] = X[col].fillna('missing').astype(str)
        else:
            try:
                X[col] = X[col].fillna(X[col].median())
            except:
                X[col] = X[col].fillna(0)

    if labels_df is None:
        return X, None, None, None
    else:
        y = labels_df[label].copy()
        categorical_features = [i for i, col in enumerate(feature_columns) if (col.startswith('c_')  or col.startswith('o_') or col.startswith('r'))]
        return X, y, feature_columns, categorical_features

def train_catboost_models(X, y, categorical_features):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=100,
        cat_features=categorical_features,
        eval_metric='Logloss',
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), plot=False)
    return model


def score_test(X, label):
    model = CatBoostClassifier()
    model.load_model('catboost_models/catboost_' + label + '.cbm')
    y_pred_proba = model.predict_proba(X)[:, 1]
    return y_pred_proba


models = {}
used_columns = {}

train_df = pd.read_csv('data/problem_train.csv', low_memory=False)
test_df = pd.read_csv('data/problem_test.csv', low_memory=False)
labels_df = pd.read_csv('data/problem_labels.csv')

preds_df = pd.DataFrame({'id': test_df['id']})
labels = labels_df.columns[1:]

for label in labels:
    X, y, feature_columns, categorical_features = prepare_features_and_targets(train_df, label, labels_df=labels_df)
    model= train_catboost_models(X, y, categorical_features)
    models[label] = model
    used_columns[label] = feature_columns

os.makedirs('catboost_models', exist_ok=True)
for label, model in models.items():
    model_path = f'catboost_models/catboost_{label}.cbm'
    model.save_model(model_path)

for label in labels:
    X, _, _, _ = prepare_features_and_targets(test_df, label, feature_columns=used_columns[label])
    preds = score_test(X, label)
    preds_df[label] = preds

preds_df.to_csv('data/problem_test_labels.csv', index=False)
