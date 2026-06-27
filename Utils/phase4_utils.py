"""
phase4_utils.py
===============
Hospital Operations & Revenue Risk Intelligence Platform
Phase 4 Utility Module — Model Evaluation, Explainability & Fairness

Imported by: Phase 4, Final Phase

Contents
--------
1.  evaluate_model()         — classification report + confusion matrix
2.  plot_confusion_matrix()  — styled confusion matrix heatmap
3.  business_metrics()       — High Risk recall, Rejected recall, revenue impact
4.  plot_feature_importance()— bar chart of top N feature importances
5.  shap_summary()           — SHAP waterfall / summary plot
6.  fairness_report()        — performance segmented by gender, city, insurer
7.  generate_model_card()    — write model card to markdown file
8.  plot_roc_curve()         — one-vs-rest ROC curves for multiclass
"""

import os
import json
from datetime import datetime

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

# ── Import shared constants ────────────────────────────────────────────────
from phase1_utils import (
    RISK_COLORS,
    STATUS_COLORS,
    ENCODING_MAPS,
    DECODING_MAPS,
    save_fig,
)
from phase3_utils import (
    MODEL_A_CLASSES,
    MODEL_B_CLASSES,
    HIGH_RISK_CLASS_IDX,
    REJECTED_CLASS_IDX,
)


# ══════════════════════════════════════════════════════════════════════════════
# 1. EVALUATE MODEL
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    class_names: list,
    split_label: str = 'Test',
    verbose: bool = True,
):
    """
    Run predictions and return a full classification report as a DataFrame.

    Parameters
    ----------
    model       : fitted sklearn estimator
    X           : pd.DataFrame  Feature matrix
    y           : pd.Series     True labels (encoded integers)
    class_names : list          Human-readable class names e.g. ['High','Low','Medium']
    split_label : str           'Train' or 'Test' — printed in header
    verbose     : bool          Print the classification report

    Returns
    -------
    y_pred   : np.ndarray  Predicted class indices
    y_proba  : np.ndarray  Predicted class probabilities (n_samples × n_classes)
    report_df: pd.DataFrame  Classification report as DataFrame

    Usage
    -----
    from phase4_utils import evaluate_model
    from phase3_utils import MODEL_A_CLASSES
    y_pred, y_proba, report = evaluate_model(model_a, X_test, y_test, MODEL_A_CLASSES)
    """
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)

    report_dict = classification_report(
        y, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).T

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {split_label} Set — Classification Report")
        print(f"{'─'*60}")
        print(classification_report(y, y_pred, target_names=class_names, zero_division=0))

    return y_pred, y_proba, report_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONFUSION MATRIX PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    class_names: list,
    title: str = 'Confusion Matrix',
    out=None,
    filename: str = 'confusion_matrix.png',
):
    """
    Plot a styled confusion matrix heatmap.

    Parameters
    ----------
    y_true      : pd.Series    True labels (encoded)
    y_pred      : np.ndarray   Predicted labels (encoded)
    class_names : list         Class label names
    title       : str          Plot title
    out         : callable     Path helper from set_output_dir() — pass to save
    filename    : str          Output filename

    Returns
    -------
    fig : matplotlib Figure

    Usage
    -----
    from phase4_utils import plot_confusion_matrix
    from phase3_utils import MODEL_A_CLASSES
    plot_confusion_matrix(y_test, y_pred, MODEL_A_CLASSES, out=out, filename='cm_a.png')
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, title_suffix in [
        (axes[0], cm,      'd',    '— Counts'),
        (axes[1], cm_norm, '.2f',  '— Normalised'),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8},
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{title} {title_suffix}')

    plt.tight_layout()
    if out:
        save_fig(out, filename)
    else:
        plt.show()

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUSINESS METRICS
# ══════════════════════════════════════════════════════════════════════════════
def business_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    test_df: pd.DataFrame = None,
    verbose: bool = True,
):
    """
    Compute business-critical metrics beyond standard accuracy.

    For Model A (visit risk):
      - High Risk recall: % of actual High Risk visits correctly flagged
      - Missing a High Risk visit = patient safety risk

    For Model B (claim outcome):
      - Rejected recall: % of actual Rejected claims correctly predicted
      - Missing a Rejection = revenue leakage

    Parameters
    ----------
    y_true     : pd.Series    True labels (encoded)
    y_pred     : np.ndarray   Predicted labels (encoded)
    model_name : str          'Model A — Visit Risk' or 'Model B — Claim Outcome'
    test_df    : pd.DataFrame Optional — used to compute revenue impact for Model B
    verbose    : bool         Print results

    Returns
    -------
    dict  Business metrics

    Usage
    -----
    from phase4_utils import business_metrics
    bm = business_metrics(y_test, y_pred, 'Model A — Visit Risk')
    """
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    metrics = {
        'model'            : model_name,
        'overall_accuracy' : float((y_pred == y_true).mean()),
        'macro_f1'         : float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1'      : float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }

    # Model A — High Risk metrics (class index 0)
    if 'Risk' in model_name or 'Model A' in model_name:
        hr_recall    = float(rec[HIGH_RISK_CLASS_IDX])
        hr_precision = float(prec[HIGH_RISK_CLASS_IDX])
        hr_f1        = float(f1[HIGH_RISK_CLASS_IDX])
        hr_support   = int(support[HIGH_RISK_CLASS_IDX])
        hr_missed    = int(((y_true == HIGH_RISK_CLASS_IDX) & (y_pred != HIGH_RISK_CLASS_IDX)).sum())

        metrics.update({
            'high_risk_recall'   : hr_recall,
            'high_risk_precision': hr_precision,
            'high_risk_f1'       : hr_f1,
            'high_risk_support'  : hr_support,
            'high_risk_missed'   : hr_missed,
        })

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  Business Metrics — {model_name}")
            print(f"{'─'*55}")
            print(f"  Overall accuracy    : {metrics['overall_accuracy']:.4f}")
            print(f"  Macro F1            : {metrics['macro_f1']:.4f}")
            print(f"  ── HIGH RISK CLASS ─────────────────────────────")
            print(f"  Recall              : {hr_recall:.4f}  "
                  f"({'⚠️ Below target' if hr_recall < 0.70 else '✅ Acceptable'})")
            print(f"  Precision           : {hr_precision:.4f}")
            print(f"  F1                  : {hr_f1:.4f}")
            print(f"  Missed High Risk    : {hr_missed} visits out of {hr_support}")
            print(f"  💡 Target: High Risk recall ≥ 0.70 for safe clinical use.")

    # Model B — Rejected metrics (class index 2)
    if 'Claim' in model_name or 'Model B' in model_name:
        rej_recall    = float(rec[REJECTED_CLASS_IDX])
        rej_precision = float(prec[REJECTED_CLASS_IDX])
        rej_f1        = float(f1[REJECTED_CLASS_IDX])
        rej_support   = int(support[REJECTED_CLASS_IDX])
        rej_missed    = int(((y_true == REJECTED_CLASS_IDX) & (y_pred != REJECTED_CLASS_IDX)).sum())

        metrics.update({
            'rejected_recall'   : rej_recall,
            'rejected_precision': rej_precision,
            'rejected_f1'       : rej_f1,
            'rejected_support'  : rej_support,
            'rejected_missed'   : rej_missed,
        })

        # Revenue impact estimate
        if test_df is not None and 'billed_amount_capped' in test_df.columns:
            avg_bill     = test_df['billed_amount_capped'].mean()
            revenue_saved = rej_missed * avg_bill
            metrics['estimated_revenue_at_risk'] = float(revenue_saved)
        else:
            metrics['estimated_revenue_at_risk'] = None

        if verbose:
            print(f"\n{'─'*55}")
            print(f"  Business Metrics — {model_name}")
            print(f"{'─'*55}")
            print(f"  Overall accuracy    : {metrics['overall_accuracy']:.4f}")
            print(f"  Macro F1            : {metrics['macro_f1']:.4f}")
            print(f"  ── REJECTED CLASS ──────────────────────────────")
            print(f"  Recall              : {rej_recall:.4f}  "
                  f"({'⚠️ Below target' if rej_recall < 0.60 else '✅ Acceptable'})")
            print(f"  Precision           : {rej_precision:.4f}")
            print(f"  F1                  : {rej_f1:.4f}")
            print(f"  Missed Rejections   : {rej_missed} claims out of {rej_support}")
            if metrics['estimated_revenue_at_risk']:
                print(f"  Revenue at risk     : ₹{metrics['estimated_revenue_at_risk']:,.0f}")
            print(f"  💡 Target: Rejected recall ≥ 0.60 to protect revenue.")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(
    model,
    features: list,
    title: str = 'Feature Importance',
    top_n: int = 15,
    out=None,
    filename: str = 'feature_importance.png',
):
    """
    Plot a horizontal bar chart of the top N feature importances.
    Works with RandomForest and GradientBoosting models.

    Parameters
    ----------
    model    : fitted sklearn tree-based estimator
    features : list  Feature names (same order used in training)
    title    : str   Plot title
    top_n    : int   Number of top features to show (default 15)
    out      : callable  Path helper
    filename : str   Output filename

    Usage
    -----
    from phase4_utils import plot_feature_importance
    plot_feature_importance(model_a, MODEL_A_FEATURES, title='Model A', out=out)
    """
    if not hasattr(model, 'feature_importances_'):
        print("⚠️  Model does not have feature_importances_. "
              "Use RandomForest or GradientBoosting.")
        return

    importances = model.feature_importances_
    feat_df = (
        pd.DataFrame({'feature': features, 'importance': importances})
        .sort_values('importance', ascending=True)
        .tail(top_n)
    )

    # Colour bars by importance magnitude
    norm = (feat_df['importance'] - feat_df['importance'].min()) / \
           (feat_df['importance'].max() - feat_df['importance'].min() + 1e-9)
    colors = [plt.cm.Blues(0.3 + 0.7 * v) for v in norm]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    bars = ax.barh(feat_df['feature'], feat_df['importance'],
                   color=colors, edgecolor='white', alpha=0.9)
    ax.set_xlabel('Importance')
    ax.set_title(f'{title} — Top {top_n} Feature Importances')

    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f'{bar.get_width():.4f}', va='center', fontsize=9)
    ax.set_xlim(0, feat_df['importance'].max() * 1.2)

    plt.tight_layout()
    if out:
        save_fig(out, filename)
    else:
        plt.show()

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. SHAP SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def shap_summary(
    model,
    X: pd.DataFrame,
    title: str = 'SHAP Summary',
    max_display: int = 15,
    out=None,
    filename: str = 'shap_summary.png',
    sample_size: int = 500,
):
    """
    Generate a SHAP summary plot showing feature impact on model output.
    Uses TreeExplainer (fast, works with RandomForest / GradientBoosting).

    Parameters
    ----------
    model       : fitted tree-based sklearn estimator
    X           : pd.DataFrame  Feature matrix (can be train or test)
    title       : str           Plot title
    max_display : int           Max features to show in SHAP plot
    out         : callable      Path helper
    filename    : str           Output filename
    sample_size : int           Rows to sample for speed (default 500)

    Usage
    -----
    from phase4_utils import shap_summary
    shap_summary(model_a, X_test, title='Model A', out=out)
    """
    try:
        import shap
    except ImportError:
        print("⚠️  SHAP not installed. Run: pip install shap")
        return

    X_sample = X.sample(min(sample_size, len(X)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For multiclass, shap_values is a list — use absolute mean across classes
    if isinstance(shap_values, list):
        shap_vals_plot = np.abs(np.array(shap_values)).mean(axis=0)
    else:
        shap_vals_plot = shap_values

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_vals_plot, X_sample,
        max_display=max_display,
        show=False,
        plot_type='bar',
    )
    plt.title(f'{title} — SHAP Feature Importance')
    plt.tight_layout()

    if out:
        save_fig(out, filename)
    else:
        plt.show()

    return shap_values


# ══════════════════════════════════════════════════════════════════════════════
# 6. FAIRNESS REPORT
# ══════════════════════════════════════════════════════════════════════════════
def fairness_report(
    test_df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    segment_cols: list,
    class_names: list,
    critical_class_idx: int,
    title: str = 'Fairness Analysis',
    out=None,
    filename: str = 'fairness_report.png',
):
    """
    Evaluate model performance segmented by demographic and operational groups.
    Flags gaps in recall for the critical class across segments.

    Parameters
    ----------
    test_df            : pd.DataFrame  Test set with original (raw) columns
    y_true             : pd.Series     True labels
    y_pred             : np.ndarray    Predicted labels
    segment_cols       : list          Columns to segment by e.g. ['gender','city','insurance_provider']
    class_names        : list          Class label names
    critical_class_idx : int           Index of the class whose recall matters most
    title              : str           Plot title prefix
    out                : callable      Path helper
    filename           : str           Output filename

    Returns
    -------
    pd.DataFrame  Fairness summary table

    Usage
    -----
    from phase4_utils import fairness_report
    from phase3_utils import HIGH_RISK_CLASS_IDX, MODEL_A_CLASSES
    fairness_report(test_df, y_test, y_pred, ['gender','city'], MODEL_A_CLASSES, HIGH_RISK_CLASS_IDX, out=out)
    """
    results = []
    critical_label = class_names[critical_class_idx]

    for col in segment_cols:
        if col not in test_df.columns:
            continue
        for segment_val in sorted(test_df[col].unique()):
            mask  = test_df[col] == segment_val
            yt    = y_true[mask]
            yp    = y_pred[mask]

            if len(yt) == 0:
                continue

            prec, rec, f1, support = precision_recall_fscore_support(
                yt, yp, zero_division=0
            )

            # Critical class recall
            if critical_class_idx < len(rec):
                crit_recall = float(rec[critical_class_idx])
            else:
                crit_recall = 0.0

            results.append({
                'segment_column' : col,
                'segment_value'  : segment_val,
                'n_records'      : int(len(yt)),
                'overall_acc'    : float((yt == yp).mean()),
                'macro_f1'       : float(f1_score(yt, yp, average='macro', zero_division=0)),
                f'{critical_label}_recall': crit_recall,
            })

    fairness_df = pd.DataFrame(results)
    if fairness_df.empty:
        print("⚠️  No fairness data to display.")
        return fairness_df

    # Plot
    n_cols = len(segment_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    recall_col = f'{critical_label}_recall'
    for ax, col in zip(axes, segment_cols):
        sub = fairness_df[fairness_df['segment_column'] == col].copy()
        if sub.empty:
            continue
        bar_colors = [
            '#C44E52' if v < 0.65 else '#DD8452' if v < 0.75 else '#55A868'
            for v in sub[recall_col]
        ]
        bars = ax.bar(sub['segment_value'].astype(str), sub[recall_col],
                      color=bar_colors, edgecolor='white', alpha=0.85)
        ax.axhline(sub[recall_col].mean(), color='black', linestyle='--',
                   linewidth=1.2, label=f'Mean: {sub[recall_col].mean():.2f}')
        ax.set_title(f'{critical_label} Recall by {col}')
        ax.set_ylabel(f'{critical_label} Recall')
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9)
        ax.tick_params(axis='x', rotation=15)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{bar.get_height():.2f}', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle(f'{title} — {critical_label} Recall by Segment', fontsize=13, fontweight='bold')
    plt.tight_layout()
    if out:
        save_fig(out, filename)
    else:
        plt.show()

    return fairness_df


# ══════════════════════════════════════════════════════════════════════════════
# 7. GENERATE MODEL CARD
# ══════════════════════════════════════════════════════════════════════════════
def generate_model_card(
    model_name: str,
    model_type: str,
    features: list,
    class_names: list,
    train_metrics: dict,
    test_metrics: dict,
    business_m: dict,
    fairness_df: pd.DataFrame = None,
    out=None,
    filename: str = 'model_card.md',
):
    """
    Write a Markdown model card documenting model performance, limitations,
    and assumptions. Following best practices from Google's Model Cards paper.

    Parameters
    ----------
    model_name     : str           e.g. 'Model A — Visit Risk Classifier'
    model_type     : str           e.g. 'Random Forest'
    features       : list          Feature names used
    class_names    : list          Class label names
    train_metrics  : dict          Output of business_metrics() on train set
    test_metrics   : dict          Output of business_metrics() on test set
    business_m     : dict          Output of business_metrics()
    fairness_df    : pd.DataFrame  Output of fairness_report()
    out            : callable      Path helper
    filename       : str           Output .md filename

    Usage
    -----
    from phase4_utils import generate_model_card
    generate_model_card('Model A', 'Random Forest', features, classes, train_m, test_m, bm, out=out)
    """
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [
        f"# Model Card — {model_name}",
        f"*Generated: {now}*",
        "",
        "## Model Details",
        f"- **Model type:** {model_type}",
        f"- **Target variable:** {class_names}",
        f"- **Number of features:** {len(features)}",
        f"- **Training split:** Time-based 80/20 (cut-off: 2025-11-08)",
        f"- **Class imbalance handling:** class_weight='balanced'",
        "",
        "## Feature List",
    ]
    for f in features:
        lines.append(f"- `{f}`")

    lines += [
        "",
        "## Performance Summary",
        "",
        "| Metric | Train | Test |",
        "|--------|-------|------|",
        f"| Overall Accuracy | {train_metrics.get('overall_accuracy', 'N/A'):.4f} | {test_metrics.get('overall_accuracy', 'N/A'):.4f} |",
        f"| Macro F1         | {train_metrics.get('macro_f1', 'N/A'):.4f} | {test_metrics.get('macro_f1', 'N/A'):.4f} |",
        f"| Weighted F1      | {train_metrics.get('weighted_f1', 'N/A'):.4f} | {test_metrics.get('weighted_f1', 'N/A'):.4f} |",
    ]

    # Model A specific
    if 'high_risk_recall' in test_metrics:
        lines += [
            f"| High Risk Recall  | {train_metrics.get('high_risk_recall','N/A'):.4f} | {test_metrics.get('high_risk_recall','N/A'):.4f} |",
            f"| High Risk Missed  | {train_metrics.get('high_risk_missed','N/A')} | {test_metrics.get('high_risk_missed','N/A')} |",
        ]

    # Model B specific
    if 'rejected_recall' in test_metrics:
        lines += [
            f"| Rejected Recall   | {train_metrics.get('rejected_recall','N/A'):.4f} | {test_metrics.get('rejected_recall','N/A'):.4f} |",
            f"| Rejected Missed   | {train_metrics.get('rejected_missed','N/A')} | {test_metrics.get('rejected_missed','N/A')} |",
        ]
        if test_metrics.get('estimated_revenue_at_risk'):
            lines.append(f"| Revenue at Risk   | — | ₹{test_metrics['estimated_revenue_at_risk']:,.0f} |")

    # Fairness summary
    if fairness_df is not None and not fairness_df.empty:
        lines += ["", "## Fairness Summary"]
        recall_col = [c for c in fairness_df.columns if '_recall' in c]
        if recall_col:
            rc = recall_col[0]
            lines.append(f"\n*{rc} across demographic segments:*\n")
            lines.append("| Segment | Value | Recall |")
            lines.append("|---------|-------|--------|")
            for _, row in fairness_df.iterrows():
                lines.append(f"| {row['segment_column']} | {row['segment_value']} | {row[rc]:.4f} |")

    lines += [
        "",
        "## Limitations & Assumptions",
        "- Trained on data from 2025-01-20 to 2025-11-07 only.",
        "- Performance may degrade if hospital operations change significantly.",
        "- Provider rejection rates are computed from historical data — new insurers will use the overall average.",
        "- Model does not account for seasonal outbreaks or policy changes post training.",
        "- Fairness analysis is limited to gender, city, and insurance_provider segments.",
        "",
        "## Retraining Strategy",
        "- Monitor prediction drift monthly using Phase 6 tools.",
        "- Retrain if macro F1 drops > 5% below baseline on fresh data.",
        "- Full retraining recommended every 6 months.",
        "",
        "## Intended Use",
        "- **Intended:** Decision support for hospital triage and finance teams.",
        "- **Not intended:** Sole basis for clinical decisions without human review.",
    ]

    content = "\n".join(lines)

    path = out(filename) if out else filename
    with open(path, 'w') as f:
        f.write(content)

    print(f"  ✅ Model card saved → {path}")
    return content


# ══════════════════════════════════════════════════════════════════════════════
# 8. ROC CURVE PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_curve(
    y_true: pd.Series,
    y_proba: np.ndarray,
    class_names: list,
    title: str = 'ROC Curves',
    out=None,
    filename: str = 'roc_curves.png',
):
    """
    Plot one-vs-rest ROC curves for each class in a multiclass classifier.

    Parameters
    ----------
    y_true      : pd.Series    True labels (encoded integers)
    y_proba     : np.ndarray   Predicted probabilities (n × n_classes)
    class_names : list         Class label names
    title       : str          Plot title
    out         : callable     Path helper
    filename    : str          Output filename

    Usage
    -----
    from phase4_utils import plot_roc_curve
    plot_roc_curve(y_test, y_proba, MODEL_A_CLASSES, title='Model A', out=out)
    """
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))
    colors    = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{title} — One-vs-Rest ROC Curves')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

    plt.tight_layout()
    if out:
        save_fig(out, filename)
    else:
        plt.show()

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys, shutil
    sys.path.insert(0, '.')
    from phase1_utils import notebook_setup, build_model_table, MODEL_A_FEATURES, MODEL_A_TARGET
    from phase3_utils  import time_split, train_model

    print("Running phase4_utils self-test...\n")

    ctx      = notebook_setup(verbose=False)
    model_df = build_model_table(ctx['df'])
    train, test = time_split(model_df, verbose=False)

    model, X_tr, y_tr = train_model(train, MODEL_A_FEATURES, MODEL_A_TARGET, verbose=False)
    X_test = test[MODEL_A_FEATURES]
    y_test = test[MODEL_A_TARGET]

    y_pred, y_proba, report = evaluate_model(model, X_test, y_test, MODEL_A_CLASSES, verbose=False)
    bm = business_metrics(y_test, y_pred, 'Model A — Visit Risk', verbose=True)
    print(f"\nBusiness metrics keys: {list(bm.keys())}")
    print("\n✅ phase4_utils self-test passed.")
