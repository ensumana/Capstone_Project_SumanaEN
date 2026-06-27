"""
phase3_utils.py
===============
Hospital Operations & Revenue Risk Intelligence Platform
Phase 3 Utility Module — Model Training & Artifact Management

Imported by: Phase 3, Phase 4, Phase 5, Phase 6

Contents
--------
1.  Paths & constants
2.  time_split()            — time-based 80/20 train/test split
3.  train_model()           — train Logistic Regression or Random Forest
4.  tune_model()            — GridSearchCV hyperparameter tuning
5.  save_model()            — save model artifact as .pkl
6.  load_model()            — load model artifact from .pkl
7.  save_feature_schema()   — save feature metadata to JSON
8.  load_feature_schema()   — load feature schema from JSON
9.  get_model_version()     — deterministic version string from timestamp
"""

import os
import json
import pickle
import hashlib
from datetime import datetime

import numpy  as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing   import LabelEncoder

# ── Import shared constants from phase1_utils ─────────────────────────────
from phase1_utils import (
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
    MODEL_A_TARGET,
    MODEL_B_TARGET,
    ENCODING_MAPS,
    DECODING_MAPS,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. PATHS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Model artifact paths — all notebooks use the same paths
MODELS_DIR   = 'models'
MODEL_A_PATH = os.path.join(MODELS_DIR, 'model_a_risk.pkl')
MODEL_B_PATH = os.path.join(MODELS_DIR, 'model_b_claim.pkl')
SCHEMA_PATH  = os.path.join(MODELS_DIR, 'feature_schema.json')

# Time-based split — earliest 80% trains, latest 20% tests
# Computed from the dataset: split falls on 2025-11-08
SPLIT_DATE = '2025-11-08'

# Class labels (in encoded order: alphabetical after LabelEncoder)
MODEL_A_CLASSES = ['High', 'Low', 'Medium']   # risk_score_encoded: 0,1,2
MODEL_B_CLASSES = ['Paid', 'Pending', 'Rejected']  # claim_status_encoded: 0,1,2

# Business-critical classes — recall on these matters most
HIGH_RISK_CLASS_IDX    = 0   # 'High' in MODEL_A_CLASSES
REJECTED_CLASS_IDX     = 2   # 'Rejected' in MODEL_B_CLASSES

# Imbalance handling: class weights for both models
MODEL_A_CLASS_WEIGHT = 'balanced'
MODEL_B_CLASS_WEIGHT = 'balanced'


# ══════════════════════════════════════════════════════════════════════════════
# 2. TIME-BASED TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def time_split(
    model_df: pd.DataFrame,
    date_col: str  = 'visit_date',
    split_date: str = SPLIT_DATE,
    verbose: bool  = True,
):
    """
    Split a model table into train and test sets by date.
    Earlier records → train, later records → test.

    Why time-based (not random)?
    In healthcare ML, models are trained on past data and predict on future
    data. Random splits leak future information into training — time splits
    simulate real-world deployment conditions correctly.

    Parameters
    ----------
    model_df   : pd.DataFrame  The model table (output of build_model_table())
    date_col   : str           Date column to split on (default 'visit_date')
    split_date : str           Cut-off date 'YYYY-MM-DD' (default SPLIT_DATE)
    verbose    : bool          Print split summary

    Returns
    -------
    train : pd.DataFrame  Earliest 80% of records
    test  : pd.DataFrame  Latest 20% of records

    Usage
    -----
    from phase3_utils import time_split
    train, test = time_split(model_df)
    """
    df = model_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    train = df[df[date_col] <  split_date].reset_index(drop=True)
    test  = df[df[date_col] >= split_date].reset_index(drop=True)

    if verbose:
        print(f"  Split date : {split_date}")
        print(f"  Train      : {len(train):,} rows  ({len(train)/len(df)*100:.1f}%)  "
              f"up to {train[date_col].max().date()}")
        print(f"  Test       : {len(test):,} rows  ({len(test)/len(df)*100:.1f}%)  "
              f"from {test[date_col].min().date()}")

    return train, test


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
def train_model(
    train: pd.DataFrame,
    features: list,
    target: str,
    model_type: str = 'random_forest',
    class_weight: str = 'balanced',
    random_state: int = 42,
    verbose: bool = True,
):
    """
    Train a classification model on the training set.

    Parameters
    ----------
    train        : pd.DataFrame  Training data
    features     : list          Feature column names
    target       : str           Target column name
    model_type   : str           'logistic_regression', 'random_forest',
                                 or 'gradient_boosting'
    class_weight : str or dict   'balanced' handles class imbalance automatically
    random_state : int           Reproducibility seed
    verbose      : bool          Print training summary

    Returns
    -------
    model        : fitted sklearn estimator
    X_train      : pd.DataFrame  Training features
    y_train      : pd.Series     Training target

    Usage
    -----
    from phase3_utils import train_model
    from phase1_utils import MODEL_A_FEATURES, MODEL_A_TARGET

    model, X_train, y_train = train_model(
        train, MODEL_A_FEATURES, MODEL_A_TARGET, model_type='random_forest'
    )
    """
    X_train = train[features]
    y_train = train[target]

    if model_type == 'logistic_regression':
        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=random_state,
            multi_class='multinomial',
            solver='lbfgs',
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose from: logistic_regression, random_forest, gradient_boosting")

    if verbose:
        print(f"  Training {model_type} on {len(X_train):,} rows × {len(features)} features...")

    model.fit(X_train, y_train)

    if verbose:
        train_score = model.score(X_train, y_train)
        print(f"  ✅ Done. Train accuracy: {train_score:.4f}")

    return model, X_train, y_train


# ══════════════════════════════════════════════════════════════════════════════
# 4. HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════
def tune_model(
    train: pd.DataFrame,
    features: list,
    target: str,
    model_type: str = 'random_forest',
    cv: int = 3,
    verbose: bool = True,
):
    """
    Run GridSearchCV to find the best hyperparameters for a model.

    Parameters
    ----------
    train      : pd.DataFrame  Training data
    features   : list          Feature column names
    target     : str           Target column name
    model_type : str           'logistic_regression' or 'random_forest'
    cv         : int           Number of cross-validation folds (default 3)
    verbose    : bool          Print best params and score

    Returns
    -------
    best_model    : fitted sklearn estimator with best params
    best_params   : dict  Best hyperparameters found
    best_score    : float Best cross-validated score

    Usage
    -----
    from phase3_utils import tune_model
    best_model, best_params, best_score = tune_model(train, MODEL_A_FEATURES, MODEL_A_TARGET)
    """
    X_train = train[features]
    y_train = train[target]

    if model_type == 'random_forest':
        base_model = RandomForestClassifier(
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth'   : [8, 10, 15],
            'min_samples_leaf': [3, 5],
        }
    elif model_type == 'logistic_regression':
        base_model = LogisticRegression(
            class_weight='balanced', max_iter=1000,
            random_state=42, multi_class='multinomial', solver='lbfgs'
        )
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
        }
    else:
        raise ValueError(f"Tuning not implemented for '{model_type}'")

    if verbose:
        print(f"  Running GridSearchCV ({cv}-fold) for {model_type}...")

    gs = GridSearchCV(
        base_model, param_grid,
        cv=cv, scoring='f1_macro', n_jobs=-1, verbose=0
    )
    gs.fit(X_train, y_train)

    if verbose:
        print(f"  ✅ Best params : {gs.best_params_}")
        print(f"  ✅ Best CV F1  : {gs.best_score_:.4f}")

    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ══════════════════════════════════════════════════════════════════════════════
# 5. SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════════
def save_model(model, path: str, verbose: bool = True):
    """
    Save a trained sklearn model to disk as a .pkl file.
    Creates the models/ directory automatically if it doesn't exist.

    Parameters
    ----------
    model   : fitted sklearn estimator
    path    : str   File path  e.g. 'models/model_a_risk.pkl'
    verbose : bool  Print confirmation

    Usage
    -----
    from phase3_utils import save_model, MODEL_A_PATH
    save_model(model_a, MODEL_A_PATH)
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    size_kb = os.path.getsize(path) / 1024
    if verbose:
        print(f"  ✅ Model saved → {path}  ({size_kb:.1f} KB)")


# ══════════════════════════════════════════════════════════════════════════════
# 6. LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_model(path: str, verbose: bool = True):
    """
    Load a trained sklearn model from a .pkl file.

    Parameters
    ----------
    path    : str   File path  e.g. 'models/model_a_risk.pkl'
    verbose : bool  Print confirmation

    Returns
    -------
    model : fitted sklearn estimator

    Usage
    -----
    from phase3_utils import load_model, MODEL_A_PATH
    model_a = load_model(MODEL_A_PATH)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            f"Run Phase 3 notebook first to train and save the model."
        )
    with open(path, 'rb') as f:
        model = pickle.load(f)
    if verbose:
        print(f"  ✅ Model loaded ← {path}  ({type(model).__name__})")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE FEATURE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
def save_feature_schema(
    train: pd.DataFrame,
    features: list,
    path: str = SCHEMA_PATH,
    verbose: bool = True,
):
    """
    Save a JSON file describing all model features — names, dtypes, ranges.
    Used in Phase 5 (API validation) and Phase 6 (drift detection).

    Parameters
    ----------
    train    : pd.DataFrame  Training data (used to compute ranges)
    features : list          Feature column names
    path     : str           Output JSON path (default SCHEMA_PATH)
    verbose  : bool          Print confirmation

    Usage
    -----
    from phase3_utils import save_feature_schema, SCHEMA_PATH
    from phase1_utils import MODEL_A_FEATURES
    save_feature_schema(train, MODEL_A_FEATURES)
    """
    schema = {
        'created_at': datetime.now().isoformat(),
        'n_features' : len(features),
        'features'   : {}
    }

    for feat in features:
        col = train[feat]
        schema['features'][feat] = {
            'dtype'  : str(col.dtype),
            'min'    : float(col.min()),
            'max'    : float(col.max()),
            'mean'   : float(col.mean()),
            'std'    : float(col.std()),
            'nulls'  : int(col.isnull().sum()),
        }

    schema['model_a_target']   = MODEL_A_TARGET
    schema['model_b_target']   = MODEL_B_TARGET
    schema['model_a_classes']  = MODEL_A_CLASSES
    schema['model_b_classes']  = MODEL_B_CLASSES
    schema['split_date']       = SPLIT_DATE

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(schema, f, indent=2)

    if verbose:
        print(f"  ✅ Feature schema saved → {path}  ({len(features)} features)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. LOAD FEATURE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
def load_feature_schema(path: str = SCHEMA_PATH, verbose: bool = True):
    """
    Load the feature schema JSON file.

    Returns
    -------
    dict  Feature schema with names, dtypes, ranges

    Usage
    -----
    from phase3_utils import load_feature_schema
    schema = load_feature_schema()
    print(schema['features']['age'])
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature schema not found at '{path}'. "
            f"Run Phase 3 notebook first."
        )
    with open(path, 'r') as f:
        schema = json.load(f)
    if verbose:
        print(f"  ✅ Schema loaded ← {path}  ({schema['n_features']} features)")
    return schema


# ══════════════════════════════════════════════════════════════════════════════
# 9. MODEL VERSION STRING
# ══════════════════════════════════════════════════════════════════════════════
def get_model_version(prefix: str = 'v') -> str:
    """
    Generate a deterministic model version string based on the current date.
    Used in Phase 5 prediction logs and Phase 6 audit trails.

    Returns
    -------
    str  e.g. 'v20251108'

    Usage
    -----
    from phase3_utils import get_model_version
    version = get_model_version()  # 'v20251108'
    """
    return f"{prefix}{datetime.now().strftime('%Y%m%d')}"


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys, shutil
    sys.path.insert(0, '.')
    from phase1_utils import notebook_setup, build_model_table

    print("Running phase3_utils self-test...\n")

    ctx      = notebook_setup(verbose=False, load_db=True, load_merged=True)
    model_df = build_model_table(ctx['df'])

    train, test = time_split(model_df)

    model, X_tr, y_tr = train_model(train, MODEL_A_FEATURES, MODEL_A_TARGET)
    save_model(model, 'models/_test_model_a.pkl')
    loaded = load_model('models/_test_model_a.pkl')
    save_feature_schema(train, MODEL_A_FEATURES, path='models/_test_schema.json')
    schema = load_feature_schema(path='models/_test_schema.json')

    print(f"\nVersion string: {get_model_version()}")

    # cleanup
    os.remove('models/_test_model_a.pkl')
    os.remove('models/_test_schema.json')
    try: os.rmdir('models')
    except: pass

    print("\n✅ phase3_utils self-test passed.")
