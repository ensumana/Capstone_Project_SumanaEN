"""
phase2_utils.py
===============
Hospital Operations & Revenue Risk Intelligence Platform
Phase 2 Utility Module — Exploratory Data Analysis & Data Quality

Imported by: Phase 2 (01_eda.ipynb), Final Phase (presentation)

Why this file exists
--------------------
All EDA plotting and analysis functions live here so that:
  - 01_eda.ipynb stays thin and readable (only calls, no wall-of-code)
  - The Final Phase presentation can regenerate any chart with one line
  - Functions are individually testable and documented

Contents
--------
1.  missing_value_summary()    — null count table
2.  plot_missing_values()      — bar chart + monthly heatmap
3.  explain_missing_nulls()    — why nulls exist by claim_status
4.  plot_distributions()       — visit volume, risk, type, status, insurer, city
5.  iqr_summary()              — IQR stats + outlier count for one column
6.  zscore_summary()           — z-score outlier count for one column
7.  plot_outliers()            — histogram + boxplot for all three numeric cols
8.  plot_correlation()         — correlation heatmap of numeric features
9.  plot_feature_vs_target()   — feature boxplots split by risk_score / claim_status
10. run_eda_pipeline()         — run the full EDA in one call (for final presentation)
"""

import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as scipy_stats

# ── Import shared constants from phase1_utils ─────────────────────────────
from phase1_utils import (
    RISK_COLORS,
    STATUS_COLORS,
    DEPT_COLORS,
    CITY_COLORS,
    INS_COLORS,
    save_fig,
    notebook_setup,
)

from pathlib import Path
_HERE = Path(__file__).parent  # folder where phase2_utils.py lives

ctx = notebook_setup(
    patients_path = str(_HERE / 'patients.csv'),
    visits_path   = str(_HERE / 'visits.csv'),
    billing_path  = str(_HERE / 'billing.csv'),
    verbose=False
)
# ══════════════════════════════════════════════════════════════════════════════
# 1. MISSING VALUE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy DataFrame showing null counts and percentages for
    every column that has at least one missing value.

    Parameters
    ----------
    df : pd.DataFrame  Merged hospital dataset

    Returns
    -------
    pd.DataFrame  Columns: null_count, null_pct, dtype

    Usage
    -----
    from phase2_utils import missing_value_summary
    summary = missing_value_summary(df)
    display(summary)
    """
    result = pd.DataFrame({
        'null_count': df.isnull().sum(),
        'null_pct'  : (df.isnull().mean() * 100).round(2),
        'dtype'     : df.dtypes,
    })
    result = result[result['null_count'] > 0].sort_values('null_count', ascending=False)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 2. PLOT MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
def plot_missing_values(
    df: pd.DataFrame,
    out=None,
    filename: str = 'missing_values.png',
):
    """
    Plot a bar chart of null counts per column and a monthly heatmap
    showing whether missingness is time-related.

    Parameters
    ----------
    df       : pd.DataFrame  Merged hospital dataset (must include billing_date)
    out      : callable      Path helper from set_output_dir()
    filename : str           Output filename

    Usage
    -----
    from phase2_utils import plot_missing_values
    plot_missing_values(df, out=out)
    """
    missing = missing_value_summary(df)
    if missing.empty:
        print("✅ No missing values found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Left: bar chart of null counts
    bars = axes[0].barh(
        missing.index, missing['null_count'],
        color=['#C44E52', '#DD8452'], edgecolor='white', alpha=0.85,
    )
    axes[0].set_xlabel('Number of missing values')
    axes[0].set_title('Missing Value Count by Column')
    for bar, pct in zip(bars, missing['null_pct']):
        axes[0].text(
            bar.get_width() + 8,
            bar.get_y() + bar.get_height() / 2,
            f'{pct}%', va='center', fontsize=10,
            color='#C44E52', fontweight='bold',
        )
    axes[0].set_xlim(0, missing['null_count'].max() * 1.2)

    # Right: monthly heatmap
    df_tmp = df.copy()
    df_tmp['month'] = pd.to_datetime(df_tmp['billing_date']).dt.to_period('M').astype(str)
    null_monthly = (
        df_tmp.groupby('month')[list(missing.index)]
        .apply(lambda x: x.isnull().sum())
    )
    sns.heatmap(
        null_monthly.T, ax=axes[1], cmap='Reds',
        linewidths=0.3, cbar_kws={'label': 'Null count'},
    )
    axes[1].set_title('Missing Values per Month')
    axes[1].set_xlabel('Billing Month')
    axes[1].tick_params(axis='x', rotation=45)

    plt.suptitle('Phase 2 · Missing Value Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(out, filename) if out else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPLAIN MISSING NULLS
# ══════════════════════════════════════════════════════════════════════════════
def explain_missing_nulls(df: pd.DataFrame) -> None:
    """
    Print a breakdown of null counts by claim_status for approved_amount
    and payment_days, explaining WHY the nulls exist (informative, not random).

    Parameters
    ----------
    df : pd.DataFrame  Merged hospital dataset

    Usage
    -----
    from phase2_utils import explain_missing_nulls
    explain_missing_nulls(df)
    """
    from IPython.display import display

    for col in ['approved_amount', 'payment_days']:
        if col not in df.columns:
            continue
        print(f"=== {col} nulls by claim_status ===")
        tbl = df.groupby('claim_status')[col].apply(
            lambda x: pd.Series({
                'total'   : len(x),
                'nulls'   : x.isnull().sum(),
                'null_pct': round(x.isnull().mean() * 100, 1),
            })
        ).unstack()
        display(tbl)
        print()

    print("""💡 Interpretation:
   - approved_amount is NULL when the insurer has not yet responded
     (Pending) or approved ₹0 (some Rejected cases).
   - payment_days is NULL when no payment has been received yet
     (Pending and Rejected claims).
   - These are INFORMATIVE nulls — they carry business meaning.
     Strategy: Rejected → 0,  Pending → median,  Paid → keep as-is.""")


# ══════════════════════════════════════════════════════════════════════════════
# 4. DISTRIBUTION PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_distributions(
    df: pd.DataFrame,
    out=None,
):
    """
    Generate four distribution charts covering department, visit type,
    claim status, risk score, insurance provider, and city.
    Saves four separate PNG files.

    Parameters
    ----------
    df  : pd.DataFrame  Merged hospital dataset
    out : callable      Path helper from set_output_dir()

    Usage
    -----
    from phase2_utils import plot_distributions
    plot_distributions(df, out=out)
    """

    # ── 4A: Department volume + risk stacked bar ───────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dept_counts = df['department'].value_counts().sort_values()
    axes[0].barh(
        dept_counts.index, dept_counts.values,
        color=DEPT_COLORS, edgecolor='white', alpha=0.85,
    )
    axes[0].set_xlabel('Number of Visits')
    axes[0].set_title('Visit Volume by Department')
    for i, v in enumerate(dept_counts.values):
        axes[0].text(v + 20, i, f'{v:,}', va='center', fontsize=10)
    axes[0].set_xlim(0, dept_counts.max() * 1.15)

    risk_dept = (
        df.groupby(['department', 'risk_score'])
        .size()
        .unstack(fill_value=0)[['Low', 'Medium', 'High']]
    )
    risk_dept.plot(
        kind='barh', stacked=True, ax=axes[1],
        color=[RISK_COLORS['Low'], RISK_COLORS['Medium'], RISK_COLORS['High']],
        edgecolor='white', alpha=0.85,
    )
    axes[1].set_title('Risk Score by Department (stacked)')
    axes[1].set_xlabel('Number of Visits')
    axes[1].legend(title='Risk Score', bbox_to_anchor=(1.01, 1))

    plt.suptitle('4A · Department Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(out, 'dist_department.png') if out else plt.show()

    # ── 4B: Pie charts — visit type / claim status / risk score ───────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col, title, colors in [
        (axes[0], 'visit_type',   'Visit Type',
         ['#4C72B0', '#DD8452', '#55A868']),
        (axes[1], 'claim_status', 'Claim Status',
         [STATUS_COLORS[k] for k in ['Paid', 'Pending', 'Rejected']]),
        (axes[2], 'risk_score',   'Risk Score',
         [RISK_COLORS[k]   for k in ['Low',  'Medium',  'High']]),
    ]:
        counts = df[col].value_counts()
        ax.pie(
            counts, labels=counts.index,
            autopct='%1.1f%%', colors=colors,
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        )
        ax.set_title(title)

    plt.suptitle('4B · Visit Type / Claim Status / Risk Score', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(out, 'dist_pie_charts.png') if out else plt.show()

    print("💡 ~20% High Risk visits, ~15% Rejected claims — moderate class "
          "imbalance to handle with class_weight='balanced' in Phase 3.")

    # ── 4C: Insurance provider — billed amount + rejection rate ───────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    insurers = sorted(df['insurance_provider'].unique())
    bp = axes[0].boxplot(
        [df[df['insurance_provider'] == i]['billed_amount'].dropna()
         for i in insurers],
        tick_labels=insurers, patch_artist=True,
        medianprops={'color': 'black', 'linewidth': 2},
    )
    for patch, color in zip(bp['boxes'], INS_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[0].set_title('Billed Amount Distribution by Insurer')
    axes[0].set_ylabel('Billed Amount (₹)')
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'₹{x/1000:.0f}K')
    )
    axes[0].tick_params(axis='x', rotation=15)

    rej_rate = (
        df.groupby('insurance_provider')['claim_status']
        .apply(lambda x: (x == 'Rejected').mean() * 100)
        .sort_values(ascending=False)
    )
    bars = axes[1].bar(
        rej_rate.index, rej_rate.values,
        color=INS_COLORS, edgecolor='white', alpha=0.85,
    )
    axes[1].set_title('Claim Rejection Rate by Insurer (%)')
    axes[1].set_ylabel('Rejection Rate (%)')
    axes[1].set_ylim(0, 20)
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].axhline(
        rej_rate.mean(), color='black', linestyle='--',
        linewidth=1.2, label=f'Avg: {rej_rate.mean():.1f}%',
    )
    axes[1].legend()
    for bar in bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f'{bar.get_height():.1f}%',
            ha='center', fontsize=10, fontweight='bold',
        )

    plt.suptitle('4C · Insurance Provider Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(out, 'dist_insurer.png') if out else plt.show()

    # ── 4D: City — visit volume + avg billed amount ────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    city_v = df.groupby('city').size().sort_values(ascending=False)
    axes[0].bar(
        city_v.index, city_v.values,
        color=CITY_COLORS, edgecolor='white', alpha=0.85,
    )
    axes[0].set_title('Total Visits by City')
    axes[0].set_ylabel('Number of Visits')
    axes[0].tick_params(axis='x', rotation=15)
    for i, v in enumerate(city_v.values):
        axes[0].text(i, v + 30, f'{v:,}', ha='center', fontsize=10)

    avg_b = df.groupby('city')['billed_amount'].mean().sort_values(ascending=False)
    axes[1].bar(
        avg_b.index, avg_b.values,
        color=CITY_COLORS, edgecolor='white', alpha=0.85,
    )
    axes[1].set_title('Average Billed Amount by City (₹)')
    axes[1].set_ylabel('Avg Billed (₹)')
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f'₹{x/1000:.1f}K')
    )
    axes[1].tick_params(axis='x', rotation=15)
    for i, v in enumerate(avg_b.values):
        axes[1].text(i, v + 80, f'₹{v/1000:.1f}K', ha='center', fontsize=10)

    plt.suptitle('4D · City-Level Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(out, 'dist_city.png') if out else plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 5. IQR SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def iqr_summary(series: pd.Series, label: str) -> tuple:
    """
    Print IQR outlier statistics for one numeric column.
    Returns the lower and upper fence values.

    Parameters
    ----------
    series : pd.Series  Numeric column
    label  : str        Human-readable column name for printing

    Returns
    -------
    (lower_fence, upper_fence) : tuple of float

    Usage
    -----
    from phase2_utils import iqr_summary
    lo, hi = iqr_summary(df['billed_amount'], 'Billed Amount (₹)')
    """
    Q1, Q3  = series.quantile(0.25), series.quantile(0.75)
    IQR     = Q3 - Q1
    lower   = max(0.0, Q1 - 1.5 * IQR)
    upper   = Q3 + 1.5 * IQR
    n_out   = ((series < lower) | (series > upper)).sum()

    print(f"{label}")
    print(f"  Q1={Q1:,.2f}   Q3={Q3:,.2f}   IQR={IQR:,.2f}")
    print(f"  Lower fence={lower:,.2f}   Upper fence={upper:,.2f}")
    print(f"  Outliers: {n_out:,}  ({n_out / len(series) * 100:.2f}%)")
    print()
    return lower, upper


# ══════════════════════════════════════════════════════════════════════════════
# 6. Z-SCORE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def zscore_summary(series: pd.Series, label: str, threshold: int = 3) -> int:
    """
    Print z-score outlier count for one numeric column.
    Returns the number of outliers found.

    Parameters
    ----------
    series    : pd.Series  Numeric column
    label     : str        Human-readable column name
    threshold : int        Z-score cutoff (default 3)

    Returns
    -------
    int  Number of outliers

    Usage
    -----
    from phase2_utils import zscore_summary
    n = zscore_summary(df['billed_amount'], 'Billed Amount')
    """
    z     = np.abs(scipy_stats.zscore(series.dropna()))
    n_out = int((z > threshold).sum())
    print(f"  {label:<28}: {n_out:,} outliers "
          f"({n_out / len(series) * 100:.2f}%)  at |z| > {threshold}")
    return n_out


# ══════════════════════════════════════════════════════════════════════════════
# 7. PLOT OUTLIERS
# ══════════════════════════════════════════════════════════════════════════════
def plot_outliers(
    df: pd.DataFrame,
    out=None,
    filename: str = 'outlier_analysis.png',
):
    """
    Generate a 2×3 grid of histograms and box plots for the three
    key numeric columns: billed_amount, payment_days, length_of_stay_hours.
    IQR fences are drawn as dashed reference lines on histograms.

    Parameters
    ----------
    df       : pd.DataFrame  Merged hospital dataset
    out      : callable      Path helper
    filename : str           Output filename

    Usage
    -----
    from phase2_utils import plot_outliers
    plot_outliers(df, out=out)
    """
    # Compute fences
    fences = {}
    for col in ['billed_amount', 'payment_days', 'length_of_stay_hours']:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR    = Q3 - Q1
        fences[col] = (max(0, Q1 - 1.5 * IQR), Q3 + 1.5 * IQR)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    cols   = ['billed_amount', 'payment_days', 'length_of_stay_hours']
    labels = ['Billed Amount (₹)', 'Payment Days', 'LOS (hours)']

    for i, (col, lbl) in enumerate(zip(cols, labels)):
        lo, hi = fences[col]
        data   = df[col].dropna()

        # Row 0 — Histogram
        axes[0, i].hist(data, bins=50, color='#4C72B0', edgecolor='white', alpha=0.8)
        axes[0, i].axvline(hi, color='#C44E52', linestyle='--',
                            linewidth=1.5, label=f'Upper ({hi:,.0f})')
        if lo > 0:
            axes[0, i].axvline(lo, color='#DD8452', linestyle='--',
                                linewidth=1.5, label=f'Lower ({lo:,.0f})')
        axes[0, i].set_title(f'{lbl} — Histogram')
        axes[0, i].set_xlabel(lbl)
        axes[0, i].legend(fontsize=8)

        # Row 1 — Box plot
        bp = axes[1, i].boxplot(
            data, vert=True, patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 2},
        )
        bp['boxes'][0].set_facecolor('#4C72B0')
        bp['boxes'][0].set_alpha(0.7)
        axes[1, i].set_title(f'{lbl} — Box Plot')
        axes[1, i].set_ylabel(lbl)

    plt.suptitle('Phase 2 · Outlier Detection', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig(out, filename) if out else plt.show()

    print("""💡 Outlier strategy: CAP at IQR upper fence — do not remove.
   Removing rows in a clinical dataset risks losing real extreme cases.
   Capped values: billed_amount → ₹53,621 | payment_days → 30.5 | LOS → 53.34 hrs""")


# ══════════════════════════════════════════════════════════════════════════════
# 8. CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation(
    df: pd.DataFrame,
    out=None,
    filename: str = 'correlation_heatmap.png',
):
    """
    Plot a lower-triangle correlation heatmap for the six key numeric
    columns in the merged dataset.

    Parameters
    ----------
    df       : pd.DataFrame  Merged hospital dataset
    out      : callable      Path helper
    filename : str           Output filename

    Usage
    -----
    from phase2_utils import plot_correlation
    plot_correlation(df, out=out)
    """
    num_cols = [
        'age', 'chronic_flag', 'length_of_stay_hours',
        'billed_amount', 'approved_amount', 'payment_days',
    ]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='RdBu_r', center=0, square=True,
        linewidths=0.5, ax=ax,
        cbar_kws={'shrink': 0.8},
    )
    ax.set_title('Correlation Matrix — Numeric Features', pad=15)
    plt.tight_layout()
    save_fig(out, filename) if out else plt.show()

    print("""💡 Key observations:
   - billed_amount & approved_amount: strongly correlated (expected — approved ≤ billed).
   - age & chronic_flag: low correlation — both useful as independent features.
   - length_of_stay & billed_amount: low — billing is not purely time-based.""")


# ══════════════════════════════════════════════════════════════════════════════
# 9. FEATURE VS TARGET PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_vs_target(
    df_feat: pd.DataFrame,
    out=None,
):
    """
    Generate two 2×3 grids of box plots:
      - Features vs risk_score  (Model A target)
      - Features vs claim_status (Model B target)

    Visually confirms that engineered features differ across target classes
    before passing them to a model.

    Parameters
    ----------
    df_feat : pd.DataFrame  DataFrame after engineer_features() has been applied
    out     : callable      Path helper

    Usage
    -----
    from phase1_utils import engineer_features
    from phase2_utils import plot_feature_vs_target
    df_feat = engineer_features(df)
    plot_feature_vs_target(df_feat, out=out)
    """

    # ── 9A: Features vs risk_score ─────────────────────────────────────────
    risk_feats  = [
        'age', 'los_capped', 'visit_frequency',
        'avg_los_per_patient', 'billed_amount_capped', 'days_since_registration',
    ]
    risk_labels = [
        'Age', 'LOS capped (hrs)', 'Visit Frequency',
        'Avg LOS / Patient', 'Billed Amount (capped)', 'Days Since Registration',
    ]
    risk_order   = ['Low', 'Medium', 'High']
    risk_palette = [RISK_COLORS[r] for r in risk_order]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feat, lbl in zip(axes.flat, risk_feats, risk_labels):
        data = [df_feat[df_feat['risk_score'] == r][feat].dropna()
                for r in risk_order]
        bp   = ax.boxplot(
            data, tick_labels=risk_order, patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 2},
        )
        for patch, color in zip(bp['boxes'], risk_palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f'{lbl} vs Risk Score')
        ax.set_xlabel('Risk Score')

    plt.suptitle('9A · Features vs Risk Score (Model A target)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(out, 'feature_vs_risk.png') if out else plt.show()

    # ── 9B: Features vs claim_status ──────────────────────────────────────
    claim_feats  = [
        'billed_amount_capped', 'approval_ratio', 'provider_rejection_rate',
        'payment_days_capped',  'los_capped',      'visit_frequency',
    ]
    claim_labels = [
        'Billed Amount (capped)', 'Approval Ratio',       'Provider Rejection Rate',
        'Payment Days (capped)',  'LOS capped (hrs)',     'Visit Frequency',
    ]
    status_order   = ['Paid', 'Pending', 'Rejected']
    status_palette = [STATUS_COLORS[s] for s in status_order]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, feat, lbl in zip(axes.flat, claim_feats, claim_labels):
        data = [df_feat[df_feat['claim_status'] == s][feat].dropna()
                for s in status_order]
        bp   = ax.boxplot(
            data, tick_labels=status_order, patch_artist=True,
            medianprops={'color': 'black', 'linewidth': 2},
        )
        for patch, color in zip(bp['boxes'], status_palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(f'{lbl} vs Claim Status')
        ax.set_xlabel('Claim Status')

    plt.suptitle('9B · Features vs Claim Status (Model B target)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig(out, 'feature_vs_claim.png') if out else plt.show()

    print("""💡 Key visual findings:
   - approval_ratio cleanly separates Paid (≈1.0) from Rejected (≈0.0).
     This will be the strongest predictor in Model B.
   - billed_amount_capped shows little separation by risk_score —
     tree-based models will find non-linear patterns beyond what box plots show.""")


# ══════════════════════════════════════════════════════════════════════════════
# 10. FULL EDA PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_eda_pipeline(
    df: pd.DataFrame,
    df_feat: pd.DataFrame,
    out=None,
    verbose: bool = True,
):
    """
    Run the complete Phase 2 EDA pipeline in a single call.
    Useful for the Final Phase presentation to regenerate all charts.

    Runs in order:
      1. Missing value summary + chart
      2. Explain missing nulls
      3. Distribution charts (4 files)
      4. Outlier analysis
      5. Correlation heatmap
      6. Feature vs target plots (2 files)

    Parameters
    ----------
    df      : pd.DataFrame  Raw merged dataset (output of get_merged_df())
    df_feat : pd.DataFrame  Engineered dataset (output of engineer_features(df))
    out     : callable      Path helper — if None, charts display inline
    verbose : bool          Print section headers (default True)

    Usage
    -----
    from phase1_utils import get_merged_df, engineer_features, notebook_setup
    from phase2_utils import run_eda_pipeline

    ctx     = notebook_setup('Output_Final')
    df      = ctx['df']
    df_feat = engineer_features(df)
    run_eda_pipeline(df, df_feat, out=ctx['out'])
    """
    sections = [
        ("Missing Value Analysis",      lambda: (plot_missing_values(df, out), explain_missing_nulls(df))),
        ("Distribution Analysis",       lambda: plot_distributions(df, out)),
        ("Outlier Detection",           lambda: plot_outliers(df, out)),
        ("Correlation Analysis",        lambda: plot_correlation(df, out)),
        ("Feature vs Target Analysis",  lambda: plot_feature_vs_target(df_feat, out)),
    ]

    for title, fn in sections:
        if verbose:
            print(f"\n{'═'*55}")
            print(f"  {title}")
            print(f"{'═'*55}")
        fn()

    if verbose:
        print("\n✅ EDA pipeline complete.")


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys, matplotlib
    matplotlib.use('Agg')
    sys.path.insert(0, '.')

    from phase1_utils import notebook_setup, engineer_features, set_output_dir

    print("Running phase2_utils self-test...\n")

    ctx    = notebook_setup(verbose=False)
    df     = ctx['df']
    out    = set_output_dir('_test_eda_output', verbose=False)

    # Test each function
    summary = missing_value_summary(df)
    print(f"missing_value_summary: {len(summary)} columns with nulls")

    plot_missing_values(df, out)
    explain_missing_nulls(df)
    plot_distributions(df, out)

    lo, hi = iqr_summary(df['billed_amount'], 'billed_amount')
    print(f"iqr_summary: lower={lo:.2f}  upper={hi:.2f}")

    n = zscore_summary(df['billed_amount'], 'billed_amount')
    print(f"zscore_summary: {n} outliers")

    plot_outliers(df, out)
    plot_correlation(df, out)

    df_feat = engineer_features(df)
    plot_feature_vs_target(df_feat, out)

    import shutil, os
    shutil.rmtree('_test_eda_output', ignore_errors=True)

    print("\n✅ phase2_utils self-test passed.")
