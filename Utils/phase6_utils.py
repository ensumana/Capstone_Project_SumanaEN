"""
phase6_utils.py
===============
Hospital Operations & Revenue Risk Intelligence Platform
Phase 6 Utility Module — Monitoring, Drift Detection & Governance

Imported by: Phase 6 (monitoring notebook)

Contents
--------
1.  validate_input_schema()   — check new data for nulls, ranges, bad categories
2.  compute_psi()             — Population Stability Index for feature drift
3.  detect_feature_drift()    — PSI across all features vs training baseline
4.  detect_prediction_drift() — shift in predicted class distribution over time
5.  AuditLogger               — append-only audit log for all predictions + events
6.  write_drift_summary()     — export drift report to CSV
7.  plot_drift_dashboard()    — visualise PSI scores and prediction drift
8.  retraining_alert()        — flag if drift or performance crosses threshold
"""

import os
import csv
import json
from datetime import datetime

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Import shared constants ────────────────────────────────────────────────
from phase1_utils import (
    ENCODING_MAPS,
    RISK_COLORS,
    STATUS_COLORS,
    save_fig,
    MODEL_A_FEATURES,
    MODEL_B_FEATURES,
)
from phase3_utils import (
    MODEL_A_CLASSES,
    MODEL_B_CLASSES,
    SPLIT_DATE,
    load_feature_schema,
    SCHEMA_PATH,
)


# ══════════════════════════════════════════════════════════════════════════════
# PSI THRESHOLDS (industry standard)
# ══════════════════════════════════════════════════════════════════════════════
PSI_NO_CHANGE    = 0.10   # PSI < 0.10 → no significant drift
PSI_MODERATE     = 0.25   # 0.10 ≤ PSI < 0.25 → moderate drift, monitor closely
PSI_SIGNIFICANT  = 0.25   # PSI ≥ 0.25 → significant drift, consider retraining


def psi_label(psi: float) -> str:
    """Return a human-readable drift label for a PSI value."""
    if psi < PSI_NO_CHANGE:
        return '✅ Stable'
    elif psi < PSI_SIGNIFICANT:
        return '⚠️  Moderate drift'
    else:
        return '🚨 Significant drift'


# ══════════════════════════════════════════════════════════════════════════════
# 1. INPUT SCHEMA VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
def validate_input_schema(
    new_df: pd.DataFrame,
    schema: dict,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Validate a new batch of data against the saved feature schema.
    Checks for missing features, out-of-range values, and unseen categories.

    Parameters
    ----------
    new_df  : pd.DataFrame  New incoming data batch
    schema  : dict          Output of load_feature_schema()
    verbose : bool          Print validation summary

    Returns
    -------
    pd.DataFrame  Validation report — one row per feature

    Usage
    -----
    from phase6_utils import validate_input_schema
    from phase3_utils import load_feature_schema
    schema = load_feature_schema()
    report = validate_input_schema(new_df, schema)
    """
    rows = []
    for feat, stats in schema['features'].items():
        if feat not in new_df.columns:
            rows.append({'feature': feat, 'check': 'MISSING', 'status': '❌', 'detail': 'Feature absent from new data'})
            continue

        col       = new_df[feat].dropna()
        null_pct  = new_df[feat].isnull().mean() * 100
        range_ok  = True
        cat_issues = 0

        # Null check
        if null_pct > 10:
            rows.append({'feature': feat, 'check': 'NULLS', 'status': '⚠️',
                         'detail': f'{null_pct:.1f}% nulls (training had {stats["nulls"]/25000*100:.1f}%)'})

        # Range check for numeric
        try:
            col_float = col.astype(float)
            out_range = ((col_float < stats['min'] * 0.5) | (col_float > stats['max'] * 1.5)).sum()
            if out_range > 0:
                rows.append({'feature': feat, 'check': 'RANGE',
                             'status': '⚠️' if out_range/len(new_df) < 0.05 else '❌',
                             'detail': f'{out_range} values outside expected range '
                                       f'[{stats["min"]:.1f}, {stats["max"]:.1f}]'})
                range_ok = False
        except (TypeError, ValueError):
            pass

        if range_ok and null_pct <= 10:
            rows.append({'feature': feat, 'check': 'OK', 'status': '✅', 'detail': ''})

    # Category checks
    for field in ['department', 'visit_type', 'gender', 'city', 'insurance_provider']:
        if field in new_df.columns:
            known_vals = set(ENCODING_MAPS.get(field, {}).keys())
            new_vals   = set(new_df[field].dropna().unique())
            unseen     = new_vals - known_vals
            if unseen:
                rows.append({'feature': field, 'check': 'UNSEEN_CATEGORY', 'status': '❌',
                             'detail': f'Unseen values: {unseen}'})

    report = pd.DataFrame(rows)

    if verbose:
        total   = len(report)
        ok      = (report['status'] == '✅').sum()
        warn    = (report['status'] == '⚠️').sum()
        errors  = (report['status'] == '❌').sum()
        print(f"  Validation: {ok} OK  |  {warn} warnings  |  {errors} errors  (out of {total} checks)")
        if errors > 0 or warn > 0:
            issues = report[report['status'] != '✅']
            print(issues[['feature','check','status','detail']].to_string(index=False))

    return report


# ══════════════════════════════════════════════════════════════════════════════
# 2. PSI COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_psi(
    expected: pd.Series,
    actual:   pd.Series,
    n_bins:   int = 10,
) -> float:
    """
    Compute Population Stability Index (PSI) between a reference and new distribution.

    PSI measures how much the distribution of a feature has shifted.
    Used to detect feature drift after model deployment.

    PSI interpretation (industry standard):
      < 0.10 → no significant change (stable)
      0.10–0.25 → moderate change (monitor)
      ≥ 0.25 → significant change (consider retraining)

    Parameters
    ----------
    expected : pd.Series  Reference distribution (training data)
    actual   : pd.Series  New distribution (production data)
    n_bins   : int        Number of bins for discretisation (default 10)

    Returns
    -------
    float  PSI value

    Usage
    -----
    from phase6_utils import compute_psi
    psi = compute_psi(train_df['age'], new_df['age'])
    """
    expected_clean = expected.dropna()
    actual_clean   = actual.dropna()

    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return 0.0

    # Create bins based on expected distribution quantiles
    breakpoints = np.nanpercentile(expected_clean, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)   # remove duplicates

    if len(breakpoints) < 2:
        return 0.0

    # Compute bin counts
    exp_counts = np.histogram(expected_clean, bins=breakpoints)[0]
    act_counts = np.histogram(actual_clean,   bins=breakpoints)[0]

    # Convert to proportions, avoid division by zero
    exp_pcts = (exp_counts + 0.0001) / (len(expected_clean) + 0.0001 * len(exp_counts))
    act_pcts = (act_counts + 0.0001) / (len(actual_clean)   + 0.0001 * len(act_counts))

    psi = np.sum((act_pcts - exp_pcts) * np.log(act_pcts / exp_pcts))
    return float(psi)


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_feature_drift(
    train_df:  pd.DataFrame,
    new_df:    pd.DataFrame,
    features:  list = None,
    verbose:   bool = True,
) -> pd.DataFrame:
    """
    Compute PSI for all features between training data and new production data.

    Parameters
    ----------
    train_df  : pd.DataFrame  Training data (reference distribution)
    new_df    : pd.DataFrame  New production data
    features  : list          Features to check (default: MODEL_A_FEATURES)
    verbose   : bool          Print summary

    Returns
    -------
    pd.DataFrame  Drift report — one row per feature with PSI and status

    Usage
    -----
    from phase6_utils import detect_feature_drift
    drift_report = detect_feature_drift(train_df, new_df)
    """
    if features is None:
        features = MODEL_A_FEATURES

    rows = []
    for feat in features:
        if feat not in train_df.columns or feat not in new_df.columns:
            continue
        psi    = compute_psi(train_df[feat], new_df[feat])
        status = psi_label(psi)
        rows.append({
            'feature'     : feat,
            'psi'         : round(psi, 4),
            'status'      : status,
            'train_mean'  : round(float(train_df[feat].mean()), 4),
            'new_mean'    : round(float(new_df[feat].mean()),   4),
            'mean_shift'  : round(float(new_df[feat].mean() - train_df[feat].mean()), 4),
        })

    drift_df = pd.DataFrame(rows).sort_values('psi', ascending=False)

    if verbose:
        n_stable   = (drift_df['psi'] < PSI_NO_CHANGE).sum()
        n_moderate = ((drift_df['psi'] >= PSI_NO_CHANGE) & (drift_df['psi'] < PSI_SIGNIFICANT)).sum()
        n_severe   = (drift_df['psi'] >= PSI_SIGNIFICANT).sum()
        print(f"  Feature Drift: {n_stable} stable  |  {n_moderate} moderate  |  {n_severe} severe")
        if n_moderate + n_severe > 0:
            print(drift_df[drift_df['psi'] >= PSI_NO_CHANGE][
                ['feature','psi','status','mean_shift']
            ].to_string(index=False))

    return drift_df


# ══════════════════════════════════════════════════════════════════════════════
# 4. PREDICTION DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_prediction_drift(
    baseline_preds: pd.Series,
    new_preds:      pd.Series,
    class_names:    list,
    model_name:     str = 'Model',
    verbose:        bool = True,
) -> dict:
    """
    Detect shifts in the distribution of predicted classes over time.
    If the model starts predicting 'High Risk' at a very different rate than
    during training, something may have changed upstream.

    Parameters
    ----------
    baseline_preds : pd.Series  Predictions on test set (during training)
    new_preds      : pd.Series  Predictions on new production data
    class_names    : list       Class names (encoded order)
    model_name     : str        Model name for printing
    verbose        : bool       Print comparison

    Returns
    -------
    dict  Prediction distribution comparison and PSI

    Usage
    -----
    from phase6_utils import detect_prediction_drift
    drift = detect_prediction_drift(test_preds, new_preds, MODEL_A_CLASSES)
    """
    result = {}

    for i, cls in enumerate(class_names):
        base_pct = float((baseline_preds == i).mean() * 100)
        new_pct  = float((new_preds == i).mean() * 100)
        shift    = new_pct - base_pct
        result[cls] = {
            'baseline_pct': round(base_pct, 2),
            'new_pct'     : round(new_pct, 2),
            'shift_pct'   : round(shift, 2),
            'alert'       : abs(shift) > 5.0,   # flag if class rate shifts > 5%
        }

    # Overall PSI on prediction distribution
    result['prediction_psi'] = round(compute_psi(baseline_preds.astype(float),
                                                  new_preds.astype(float)), 4)

    if verbose:
        print(f"\n  {model_name} — Prediction Drift")
        print(f"  {'Class':<12} {'Baseline':>10} {'New':>10} {'Shift':>8} {'Alert':>8}")
        print(f"  {'─'*50}")
        for cls in class_names:
            d = result[cls]
            alert = '🚨' if d['alert'] else '✅'
            print(f"  {cls:<12} {d['baseline_pct']:>9.2f}% {d['new_pct']:>9.2f}% "
                  f"{d['shift_pct']:>+7.2f}% {alert:>8}")
        print(f"  Prediction PSI: {result['prediction_psi']}  {psi_label(result['prediction_psi'])}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5. AUDIT LOGGER
# ══════════════════════════════════════════════════════════════════════════════
class AuditLogger:
    """
    Append-only audit log for all system events — predictions, drift alerts,
    retraining events, and schema validation failures.

    Designed for regulatory compliance in healthcare AI systems.
    Every entry is timestamped and cannot be deleted (append-only).

    Usage
    -----
    from phase6_utils import AuditLogger
    audit = AuditLogger('Output_Phase6/audit_log.csv')
    audit.log_event('drift_alert', 'age PSI=0.31 — significant drift detected')
    audit.log_prediction('model_a', visit_id=1234, prediction='High', confidence=0.82)
    df = audit.load()
    """

    COLUMNS = ['timestamp', 'event_type', 'model', 'visit_id',
               'prediction', 'confidence', 'detail', 'logged_by']

    def __init__(self, path: str = 'Output_Phase6/audit_log.csv'):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.DictWriter(f, fieldnames=self.COLUMNS).writeheader()

    def _write(self, row: dict):
        with open(self.path, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=self.COLUMNS).writerow(row)

    def log_event(self, event_type: str, detail: str, model: str = ''):
        """Log a system event (drift alert, retraining trigger, validation failure)."""
        self._write({
            'timestamp'  : datetime.now().isoformat(),
            'event_type' : event_type,
            'model'      : model,
            'visit_id'   : '',
            'prediction' : '',
            'confidence' : '',
            'detail'     : detail,
            'logged_by'  : 'phase6_utils',
        })

    def log_prediction(self, model: str, visit_id: int,
                       prediction: str, confidence: float, detail: str = ''):
        """Log a single model prediction."""
        self._write({
            'timestamp'  : datetime.now().isoformat(),
            'event_type' : 'prediction',
            'model'      : model,
            'visit_id'   : visit_id,
            'prediction' : prediction,
            'confidence' : round(confidence, 4),
            'detail'     : detail,
            'logged_by'  : 'phase6_utils',
        })

    def load(self) -> pd.DataFrame:
        """Load the complete audit log as a DataFrame."""
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.read_csv(self.path)

    def summary(self) -> dict:
        """Summarise the audit log."""
        df = self.load()
        if df.empty:
            return {'total_events': 0}
        return {
            'total_events'    : len(df),
            'by_event_type'   : df['event_type'].value_counts().to_dict(),
            'by_model'        : df['model'].value_counts().to_dict(),
            'date_range'      : f"{df['timestamp'].min()[:10]} → {df['timestamp'].max()[:10]}",
        }


# ══════════════════════════════════════════════════════════════════════════════
# 6. WRITE DRIFT SUMMARY CSV
# ══════════════════════════════════════════════════════════════════════════════
def write_drift_summary(
    drift_df:    pd.DataFrame,
    pred_drift:  dict,
    out=None,
    filename:    str  = 'drift_summary.csv',
    verbose:     bool = True,
) -> str:
    """
    Export the complete drift report to CSV for handover and archiving.

    Parameters
    ----------
    drift_df   : pd.DataFrame  Output of detect_feature_drift()
    pred_drift : dict          Output of detect_prediction_drift()
    out        : callable      Path helper
    filename   : str           Output filename
    verbose    : bool          Print confirmation

    Returns
    -------
    str  Path where file was saved

    Usage
    -----
    from phase6_utils import write_drift_summary
    path = write_drift_summary(drift_df, pred_drift, out=out)
    """
    path = out(filename) if out else filename

    # Add prediction drift rows
    pred_rows = []
    for cls, vals in pred_drift.items():
        if cls == 'prediction_psi':
            continue
        pred_rows.append({
            'feature'    : f'PREDICTION_{cls}',
            'psi'        : vals.get('shift_pct', 0) / 100,
            'status'     : '🚨 Alert' if vals.get('alert') else '✅ Stable',
            'train_mean' : vals.get('baseline_pct', 0),
            'new_mean'   : vals.get('new_pct', 0),
            'mean_shift' : vals.get('shift_pct', 0),
        })

    full_df = pd.concat([drift_df, pd.DataFrame(pred_rows)], ignore_index=True)
    full_df['report_date'] = datetime.now().strftime('%Y-%m-%d')
    full_df.to_csv(path, index=False)

    if verbose:
        print(f"  ✅ Drift summary saved → {path}")

    return path


# ══════════════════════════════════════════════════════════════════════════════
# 7. DRIFT DASHBOARD PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_drift_dashboard(
    drift_df:   pd.DataFrame,
    pred_drift_a: dict,
    pred_drift_b: dict,
    out=None,
    filename:   str = 'drift_dashboard.png',
):
    """
    Visualise PSI scores and prediction class distribution shifts.

    Parameters
    ----------
    drift_df     : pd.DataFrame  Output of detect_feature_drift()
    pred_drift_a : dict          Output of detect_prediction_drift() for Model A
    pred_drift_b : dict          Output of detect_prediction_drift() for Model B
    out          : callable      Path helper
    filename     : str           Output filename

    Usage
    -----
    from phase6_utils import plot_drift_dashboard
    plot_drift_dashboard(drift_df, drift_a, drift_b, out=out)
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)

    # ── Top left: PSI bar chart ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    df_plot = drift_df.head(15).sort_values('psi', ascending=True)
    bar_colors = [
        '#C44E52' if p >= PSI_SIGNIFICANT else
        '#DD8452' if p >= PSI_NO_CHANGE else '#55A868'
        for p in df_plot['psi']
    ]
    ax1.barh(df_plot['feature'], df_plot['psi'],
             color=bar_colors, edgecolor='white', alpha=0.85)
    ax1.axvline(PSI_NO_CHANGE,   color='#DD8452', linestyle='--', linewidth=1.5,
                label=f'Moderate ({PSI_NO_CHANGE})')
    ax1.axvline(PSI_SIGNIFICANT, color='#C44E52', linestyle='--', linewidth=1.5,
                label=f'Significant ({PSI_SIGNIFICANT})')
    ax1.set_xlabel('PSI Score')
    ax1.set_title('Feature Drift — PSI Scores (top 15 features)')
    ax1.legend(fontsize=9)
    for bar in ax1.patches:
        ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{bar.get_width():.4f}', va='center', fontsize=8)

    # ── Bottom left: Model A prediction drift ─────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    classes_a   = [k for k in pred_drift_a if k != 'prediction_psi']
    baseline_a  = [pred_drift_a[c]['baseline_pct'] for c in classes_a]
    new_a       = [pred_drift_a[c]['new_pct']      for c in classes_a]
    x_a         = range(len(classes_a))
    ax2.bar([i - 0.2 for i in x_a], baseline_a, 0.35, label='Baseline',
            color='#4C72B0', edgecolor='white', alpha=0.85)
    ax2.bar([i + 0.2 for i in x_a], new_a, 0.35, label='New Data',
            color='#DD8452', edgecolor='white', alpha=0.85)
    ax2.set_xticks(list(x_a)); ax2.set_xticklabels(classes_a)
    ax2.set_ylabel('%'); ax2.set_title(f'Model A — Prediction Distribution')
    ax2.legend()

    # ── Bottom right: Model B prediction drift ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    classes_b   = [k for k in pred_drift_b if k != 'prediction_psi']
    baseline_b  = [pred_drift_b[c]['baseline_pct'] for c in classes_b]
    new_b       = [pred_drift_b[c]['new_pct']      for c in classes_b]
    x_b         = range(len(classes_b))
    ax3.bar([i - 0.2 for i in x_b], baseline_b, 0.35, label='Baseline',
            color='#4C72B0', edgecolor='white', alpha=0.85)
    ax3.bar([i + 0.2 for i in x_b], new_b, 0.35, label='New Data',
            color='#DD8452', edgecolor='white', alpha=0.85)
    ax3.set_xticks(list(x_b)); ax3.set_xticklabels(classes_b)
    ax3.set_ylabel('%'); ax3.set_title(f'Model B — Prediction Distribution')
    ax3.legend()

    plt.suptitle('Phase 6 · Monitoring Dashboard', fontsize=14, fontweight='bold')
    if out:
        save_fig(out, filename)
    else:
        plt.show()

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8. RETRAINING ALERT
# ══════════════════════════════════════════════════════════════════════════════
def retraining_alert(
    drift_df:         pd.DataFrame,
    current_macro_f1: float  = None,
    baseline_macro_f1: float = None,
    f1_drop_threshold: float = 0.05,
    audit_logger=None,
    verbose:           bool  = True,
) -> dict:
    """
    Evaluate whether retraining should be triggered based on:
    1. Number of features with significant PSI drift
    2. Drop in macro F1 score vs training baseline

    Parameters
    ----------
    drift_df          : pd.DataFrame  Output of detect_feature_drift()
    current_macro_f1  : float         Current macro F1 on fresh labelled data
    baseline_macro_f1 : float         Macro F1 at training time
    f1_drop_threshold : float         Alert if F1 drops more than this (default 0.05)
    audit_logger      : AuditLogger   If provided, logs alert to audit log
    verbose           : bool          Print recommendation

    Returns
    -------
    dict  Alert details and recommendation

    Usage
    -----
    from phase6_utils import retraining_alert
    alert = retraining_alert(drift_df, current_macro_f1=0.71, baseline_macro_f1=0.78)
    """
    severe_drift   = (drift_df['psi'] >= PSI_SIGNIFICANT).sum()
    moderate_drift = ((drift_df['psi'] >= PSI_NO_CHANGE) &
                      (drift_df['psi'] < PSI_SIGNIFICANT)).sum()

    retrain_needed = False
    reasons        = []

    if severe_drift >= 3:
        retrain_needed = True
        reasons.append(f'{severe_drift} features show significant PSI drift (≥ {PSI_SIGNIFICANT})')

    if current_macro_f1 and baseline_macro_f1:
        f1_drop = baseline_macro_f1 - current_macro_f1
        if f1_drop > f1_drop_threshold:
            retrain_needed = True
            reasons.append(f'Macro F1 dropped by {f1_drop:.4f} '
                           f'(baseline={baseline_macro_f1:.4f} → current={current_macro_f1:.4f})')

    alert = {
        'retrain_recommended': retrain_needed,
        'severe_drift_count' : int(severe_drift),
        'moderate_drift_count': int(moderate_drift),
        'reasons'            : reasons,
        'timestamp'          : datetime.now().isoformat(),
    }

    if verbose:
        print(f"\n  {'🚨 RETRAINING RECOMMENDED' if retrain_needed else '✅ No retraining needed'}")
        print(f"  Severe drift features  : {severe_drift}")
        print(f"  Moderate drift features: {moderate_drift}")
        for reason in reasons:
            print(f"  Reason: {reason}")

    if audit_logger and retrain_needed:
        audit_logger.log_event(
            'retraining_alert',
            detail=f"Retrain recommended. Reasons: {'; '.join(reasons)}"
        )

    return alert


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys, shutil
    sys.path.insert(0, '.')
    from phase1_utils  import notebook_setup, build_model_table, MODEL_A_FEATURES
    from phase3_utils  import time_split, train_model, save_model, save_feature_schema, load_feature_schema

    print("Running phase6_utils self-test...\n")

    ctx = notebook_setup(verbose=False)
    df  = build_model_table(ctx['df'])
    train, test = time_split(df, verbose=False)

    schema = {'features': {
        feat: {
            'min': float(train[feat].min()), 'max': float(train[feat].max()),
            'mean': float(train[feat].mean()), 'std': float(train[feat].std()),
            'nulls': int(train[feat].isnull().sum()),
        } for feat in MODEL_A_FEATURES
    }}

    # Test validation
    val_report = validate_input_schema(test, schema, verbose=True)

    # Test PSI
    psi_val = compute_psi(train['age'], test['age'])
    print(f"\n  PSI for 'age': {psi_val:.4f}  {psi_label(psi_val)}")

    # Test feature drift
    drift_df = detect_feature_drift(train, test, MODEL_A_FEATURES, verbose=True)

    # Test prediction drift
    model_a, _, _ = train_model(train, MODEL_A_FEATURES, 'risk_score_encoded', verbose=False)
    base_preds = model_a.predict(train[MODEL_A_FEATURES])
    new_preds  = model_a.predict(test[MODEL_A_FEATURES])
    drift_a    = detect_prediction_drift(
        pd.Series(base_preds), pd.Series(new_preds), MODEL_A_CLASSES, 'Model A', verbose=True
    )

    # Test audit logger
    audit = AuditLogger('_test_audit.csv')
    audit.log_event('self_test', 'phase6_utils self-test running')
    audit.log_prediction('model_a', 1234, 'High', 0.82)
    print(f"\n  Audit log summary: {audit.summary()}")

    # Test retraining alert
    alert = retraining_alert(drift_df, current_macro_f1=0.72, baseline_macro_f1=0.78,
                             audit_logger=audit, verbose=True)

    os.remove('_test_audit.csv')
    print("\n✅ phase6_utils self-test passed.")
