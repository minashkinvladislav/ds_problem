import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def calculate_iv(feature_series, target_series, feature_type=None, bins=10):
    feature_copy = feature_series.copy()
    feature_copy = feature_copy.fillna('missing')

    if feature_type == 'numerical' and len(feature_copy.unique()) > bins:
        non_nan_mask = feature_copy != 'missing'
        if non_nan_mask.sum() > 0:
            feature_copy[non_nan_mask] = pd.qcut(feature_series[non_nan_mask], q=bins, duplicates='drop', labels=False)

    crosstab = pd.crosstab(feature_copy, target_series)
    good_pct = np.maximum(crosstab[0] / crosstab[0].sum(), 1e-6)
    bad_pct = np.maximum(crosstab[1] / crosstab[1].sum(), 1e-6)

    total_iv = ((bad_pct - good_pct) * np.log(bad_pct / good_pct)).sum()
    return total_iv


def select_features_by_iv(df, target, min_non_nan_ratio, min_iv, type):
    selected_features = []

    for col in df.columns:
        if (df[col].count() / len(df[col])) < min_non_nan_ratio:
            continue
        iv_score = calculate_iv(df[col], target, type)
        if iv_score >= min_iv and not np.isinf(iv_score):
            selected_features.append(col)
    return selected_features


def select_features(features_df, labels_df, label, min_non_nan_ratio=0.85, min_iv=0.1):
    all_selected_features = []
    target = labels_df[label]
    name_to_range = {
        'n': range(0, 116),
        'o': range(116, 327),
        'c': range(327, 1378)
    }
    types = ['numerical', 'categorical', 'categorical']

    for name, type in zip(name_to_range.keys(), types):
        features = [f'{name}_{i:04d}' for i in name_to_range[name] if f'{name}_{i:04d}' in features_df.columns]
        selected_features = select_features_by_iv(features_df[features], target, min_non_nan_ratio, min_iv, type)
        all_selected_features.extend(selected_features)

    all_selected_features.append('release')
    return all_selected_features
