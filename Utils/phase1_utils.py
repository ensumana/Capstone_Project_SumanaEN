"""
phase1_utils.py
===============
Hospital Operations & Revenue Risk Intelligence Platform
Shared Utility Module — imported by ALL phase notebooks (Phase 1 through Phase 6)

Why this file exists
--------------------
Jupyter notebooks (.ipynb) are JSON files, not Python modules.
Python's import system cannot read them directly.
All reusable logic — setup, helpers, constants, encoding maps — lives here
so every notebook imports it with one line:

    from phase1_utils import setup_database, run_query, RISK_COLORS

Update this file once → all notebooks pick up the change automatically.

Contents
--------
1.  Imports & pandas display configuration
2.  Plot style configuration
3.  Colour palettes & constants
4.  Encoding maps (categorical → numeric, fixed across all phases)
5.  Provider rejection rates (computed from training data)
6.  setup_database()  — load CSVs, build SQLite DB, indexes, views
7.  run_query()       — run SQL, print labelled result, return DataFrame
8.  get_merged_df()   — return fully merged pandas DataFrame with parsed dates
9.  set_output_dir()  — create an output folder and return a path helper
10. save_fig()        — save current matplotlib figure to the output folder
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import sqlite3
import warnings

import numpy  as np
import pandas as pd
import matplotlib.pyplot    as plt
import matplotlib.ticker    as mticker
import seaborn              as sns
from scipy                  import stats
from sklearn.preprocessing  import LabelEncoder
from IPython.display        import display

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 2. PANDAS DISPLAY CONFIGURATION
# Calling configure_pandas() at the top of any notebook ensures:
#   - No scientific notation  (1.34e+08 → 134,591,163.08)
#   - All columns visible     (no '...' in wide DataFrames)
#   - Full text in cells      (no truncation with '...')
# ══════════════════════════════════════════════════════════════════════════════
def configure_pandas():
    """Apply consistent pandas display options across all notebooks."""
    pd.set_option('display.float_format', '{:,.2f}'.format)
    pd.set_option('display.max_columns',  None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width',        None)


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLOT STYLE CONFIGURATION
# Calling configure_plots() at the top of any notebook sets a consistent
# visual style used identically across Phase 1 through Phase 6.
# ══════════════════════════════════════════════════════════════════════════════
def configure_plots():
    """Apply consistent matplotlib/seaborn style across all notebooks."""
    plt.rcParams.update({
        'figure.dpi'        : 120,
        'figure.facecolor'  : 'white',
        'axes.facecolor'    : '#f8f8f8',
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        'axes.grid'         : True,
        'grid.alpha'        : 0.4,
        'grid.linestyle'    : '--',
        'font.size'         : 11,
        'axes.titlesize'    : 13,
        'axes.titleweight'  : 'bold',
        'axes.labelsize'    : 11,
    })


# ══════════════════════════════════════════════════════════════════════════════
# 4. COLOUR PALETTES
# All notebooks use these same palettes so every chart looks consistent.
# ══════════════════════════════════════════════════════════════════════════════

# Risk score colours — used in Model A charts
RISK_COLORS = {
    'Low'    : '#55A868',
    'Medium' : '#DD8452',
    'High'   : '#C44E52',
}

# Claim status colours — used in Model B charts
STATUS_COLORS = {
    'Paid'    : '#55A868',
    'Pending' : '#DD8452',
    'Rejected': '#C44E52',
}

# Department colours — 6 departments, 6 distinct colours
DEPT_COLORS = [
    '#4C72B0',  # Cardiology
    '#DD8452',  # ER
    '#55A868',  # General
    '#C44E52',  # ICU
    '#8172B3',  # Neurology
    '#937860',  # Orthopedics
]

# City colours — 6 cities
CITY_COLORS = [
    '#4C72B0',  # Bangalore
    '#DD8452',  # Chennai
    '#55A868',  # Delhi
    '#C44E52',  # Hyderabad
    '#8172B3',  # Mumbai
    '#937860',  # Pune
]

# Insurance provider colours — 4 providers
INS_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']


# ══════════════════════════════════════════════════════════════════════════════
# 5. ENCODING MAPS
# Fixed categorical → numeric maps used consistently across:
#   Phase 2 (feature engineering)
#   Phase 3 (model training)
#   Phase 5 (API input validation)
#   Phase 6 (drift monitoring)
#
# These are FIXED — never refit from data — so train and inference
# always encode identically.
# ══════════════════════════════════════════════════════════════════════════════
ENCODING_MAPS = {
    'department': {
        'Cardiology' : 0,
        'ER'         : 1,
        'General'    : 2,
        'ICU'        : 3,
        'Neurology'  : 4,
        'Orthopedics': 5,
    },
    'visit_type': {
        'ER' : 0,
        'ICU': 1,
        'OPD': 2,
    },
    'gender': {
        'F': 0,
        'M': 1,
    },
    'city': {
        'Bangalore': 0,
        'Chennai'  : 1,
        'Delhi'    : 2,
        'Hyderabad': 3,
        'Mumbai'   : 4,
        'Pune'     : 5,
    },
    'insurance_provider': {
        'CareOne'   : 0,
        'HealthPlus': 1,
        'MediCareX' : 2,
        'SecureLife': 3,
    },
    'risk_score': {
        'High'  : 0,
        'Low'   : 1,
        'Medium': 2,
    },
    'claim_status': {
        'Paid'    : 0,
        'Pending' : 1,
        'Rejected': 2,
    },
}

# Reverse maps — numeric → label (used in prediction APIs and reports)
DECODING_MAPS = {
    col: {v: k for k, v in mapping.items()}
    for col, mapping in ENCODING_MAPS.items()
}

# ── Provider rejection rates ───────────────────────────────────────────────
# Computed from full training data in Phase 2.
# Used in Phase 3 (feature), Phase 5 (API), Phase 6 (drift baseline).
PROVIDER_REJECTION_RATES = {
    'CareOne'   : 0.148655,
    'HealthPlus': 0.149678,
    'MediCareX' : 0.152480,
    'SecureLife': 0.156915,
}
_OVERALL_REJECTION_RATE = round(
    sum(PROVIDER_REJECTION_RATES.values()) / len(PROVIDER_REJECTION_RATES), 6
)


# ══════════════════════════════════════════════════════════════════════════════
# 6. DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════
def setup_database(
    patients_path: str = 'patients.csv',
    visits_path:   str = 'visits.csv',
    billing_path:  str = 'billing.csv',
    verbose:       bool = True,
):
    """
    Load the three raw CSVs into an in-memory SQLite database.
    Creates all Phase 1 indexes and views automatically.

    Parameters
    ----------
    patients_path : str  Path to patients.csv
    visits_path   : str  Path to visits.csv
    billing_path  : str  Path to billing.csv
    verbose       : bool Print progress messages (default True)

    Returns
    -------
    con      : sqlite3.Connection   Live database connection
    patients : pd.DataFrame         Raw patients table
    visits   : pd.DataFrame         Raw visits table
    billing  : pd.DataFrame         Raw billing table

    Usage
    -----
    from phase1_utils import setup_database
    con, patients, visits, billing = setup_database()
    """
    # ── Load CSVs ──────────────────────────────────────────────────────────
    patients = pd.read_csv(patients_path)
    visits   = pd.read_csv(visits_path)
    billing  = pd.read_csv(billing_path)

    if verbose:
        print(f"  patients : {patients.shape[0]:>6,} rows × {patients.shape[1]} cols")
        print(f"  visits   : {visits.shape[0]:>6,} rows × {visits.shape[1]} cols")
        print(f"  billing  : {billing.shape[0]:>6,} rows × {billing.shape[1]} cols")

    # ── Load into SQLite ───────────────────────────────────────────────────
    con = sqlite3.connect(':memory:')
    patients.to_sql('patients', con, index=False, if_exists='replace')
    visits.to_sql('visits',     con, index=False, if_exists='replace')
    billing.to_sql('billing',   con, index=False, if_exists='replace')

    cur = con.cursor()

    # ── Phase 1 indexes ────────────────────────────────────────────────────
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_visits_patient_id    ON visits  (patient_id)",
        "CREATE INDEX IF NOT EXISTS idx_visits_department    ON visits  (department)",
        "CREATE INDEX IF NOT EXISTS idx_visits_risk_score    ON visits  (risk_score)",
        "CREATE INDEX IF NOT EXISTS idx_visits_visit_date    ON visits  (visit_date)",
        "CREATE INDEX IF NOT EXISTS idx_billing_visit_id     ON billing (visit_id)",
        "CREATE INDEX IF NOT EXISTS idx_billing_claim_status ON billing (claim_status)",
        "CREATE INDEX IF NOT EXISTS idx_billing_billing_date ON billing (billing_date)",
    ]
    for sql in indexes:
        cur.execute(sql)

    # ── Phase 1 views ──────────────────────────────────────────────────────
    cur.execute("DROP VIEW IF EXISTS vw_hospital_master")
    cur.execute("""
        CREATE VIEW vw_hospital_master AS
        SELECT
            p.patient_id, p.age, p.gender, p.city,
            p.insurance_provider, p.chronic_flag, p.registration_date,
            v.visit_id, v.visit_date, v.department, v.visit_type,
            v.length_of_stay_hours, v.risk_score, v.doctor_id,
            b.bill_id, b.billed_amount, b.approved_amount,
            b.claim_status, b.payment_days, b.billing_date
        FROM patients p
        JOIN visits   v ON p.patient_id = v.patient_id
        JOIN billing  b ON v.visit_id   = b.visit_id
    """)

    cur.execute("DROP VIEW IF EXISTS vw_visits_patients")
    cur.execute("""
        CREATE VIEW vw_visits_patients AS
        SELECT
            p.patient_id, p.age, p.gender, p.city,
            p.insurance_provider, p.chronic_flag,
            v.visit_id, v.visit_date, v.department, v.visit_type,
            v.length_of_stay_hours, v.risk_score, v.doctor_id
        FROM patients p
        JOIN visits v ON p.patient_id = v.patient_id
    """)

    con.commit()

    if verbose:
        print("  ✅ SQLite DB ready — 7 indexes + 2 views created.")

    return con, patients, visits, billing


# ══════════════════════════════════════════════════════════════════════════════
# 7. SQL QUERY HELPER
# ══════════════════════════════════════════════════════════════════════════════
def run_query(con, title: str, sql: str, explanation: str = ""):
    """
    Run a SQL query against the database, print a labelled result,
    and return the DataFrame.

    Parameters
    ----------
    con         : sqlite3.Connection  Active database connection
    title       : str                 Label printed above the result table
    sql         : str                 SQL query string
    explanation : str                 Optional business insight printed below

    Returns
    -------
    pd.DataFrame  Query result

    Usage
    -----
    from phase1_utils import setup_database, run_query
    con, *_ = setup_database()
    df = run_query(con, "Top departments", "SELECT department, COUNT(*) FROM visits GROUP BY 1")
    """
    result = pd.read_sql(sql, con)
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    display(result)
    if explanation:
        print(f"\n💡 {explanation}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 8. MERGED DATAFRAME
# ══════════════════════════════════════════════════════════════════════════════
def get_merged_df(
    patients: pd.DataFrame,
    visits:   pd.DataFrame,
    billing:  pd.DataFrame,
    verbose:  bool = True,
) -> pd.DataFrame:
    """
    Merge the three raw DataFrames into a single analysis-ready DataFrame.
    Parses all date columns automatically.

    Parameters
    ----------
    patients : pd.DataFrame  Raw patients table
    visits   : pd.DataFrame  Raw visits table
    billing  : pd.DataFrame  Raw billing table
    verbose  : bool          Print shape and date range (default True)

    Returns
    -------
    pd.DataFrame  Merged DataFrame with parsed date columns

    Usage
    -----
    from phase1_utils import setup_database, get_merged_df
    con, patients, visits, billing = setup_database()
    df = get_merged_df(patients, visits, billing)
    """
    df = (
        visits
        .merge(patients, on='patient_id', how='left')
        .merge(billing,  on='visit_id',   how='left')
    )
    df['visit_date']        = pd.to_datetime(df['visit_date'])
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    df['billing_date']      = pd.to_datetime(df['billing_date'])

    if verbose:
        print(f"  Merged shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Date range   : {df['visit_date'].min().date()} → {df['visit_date'].max().date()}")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 9. OUTPUT DIRECTORY HELPER
# ══════════════════════════════════════════════════════════════════════════════
def set_output_dir(folder_name: str, verbose: bool = True):
    """
    Create the output folder if it does not exist.
    Returns a path-builder function `out(filename)` for that folder.

    Parameters
    ----------
    folder_name : str   Name of the output folder (e.g. 'Output_Phase2')
    verbose     : bool  Print the folder path (default True)

    Returns
    -------
    out : callable  Function that returns full path: out('chart.png')
                    → 'Output_Phase2/chart.png'

    Usage
    -----
    from phase1_utils import set_output_dir
    out = set_output_dir('Output_Phase3')
    plt.savefig(out('my_chart.png'))
    model_df.to_csv(out('model_table.csv'))
    """
    os.makedirs(folder_name, exist_ok=True)

    def out(filename: str) -> str:
        return os.path.join(folder_name, filename)

    if verbose:
        print(f"  ✅ Output folder ready → {os.path.abspath(folder_name)}/")

    return out


# ══════════════════════════════════════════════════════════════════════════════
# 10. FIGURE SAVE HELPER
# ══════════════════════════════════════════════════════════════════════════════
def save_fig(out, filename: str, verbose: bool = True):
    """
    Save the current matplotlib figure to the output folder and show it.

    Parameters
    ----------
    out      : callable  The path-builder returned by set_output_dir()
    filename : str       Filename including extension (e.g. 'chart.png')
    verbose  : bool      Print confirmation (default True)

    Usage
    -----
    from phase1_utils import set_output_dir, save_fig
    out = set_output_dir('Output_Phase2')
    plt.plot(...)
    save_fig(out, 'my_chart.png')
    """
    path = out(filename)
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    if verbose:
        print(f"  ✅ Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. IQR CAPPING HELPER
# ══════════════════════════════════════════════════════════════════════════════
def iqr_cap(series: pd.Series) -> pd.Series:
    """
    Cap a numeric series at its IQR upper fence (lower clamped to 0).
    Used in Phase 2 feature engineering and Phase 6 drift monitoring.

    Parameters
    ----------
    series : pd.Series  Numeric column to cap

    Returns
    -------
    pd.Series  Capped series (same index as input)
    """
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR    = Q3 - Q1
    upper  = Q3 + 1.5 * IQR
    lower  = max(0.0, Q1 - 1.5 * IQR)
    return series.clip(lower=lower, upper=upper)


# ══════════════════════════════════════════════════════════════════════════════
# 12. FEATURE ENGINEERING PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all 11 feature engineering steps to a merged hospital DataFrame.
    Used in Phase 2 (EDA), Phase 3 (model training), Phase 5 (API inference).

    Parameters
    ----------
    df : pd.DataFrame  Output of get_merged_df()

    Returns
    -------
    pd.DataFrame  Same DataFrame with 11 additional engineered columns
    """
    df = df.copy()

    # 1. visit_frequency — total visits by this patient
    df['visit_frequency'] = df.groupby('patient_id')['visit_id'].transform('count')

    # 2. avg_los_per_patient — patient-level average length of stay
    df['avg_los_per_patient'] = (
        df.groupby('patient_id')['length_of_stay_hours'].transform('mean')
    )

    # 3. provider_rejection_rate — historical rejection rate of their insurer
    df['provider_rejection_rate'] = (
        df['insurance_provider']
        .map(PROVIDER_REJECTION_RATES)
        .fillna(_OVERALL_REJECTION_RATE)
    )

    # 4. days_since_registration — days between registration and visit
    df['days_since_registration'] = (
        (df['visit_date'] - df['registration_date']).dt.days.clip(lower=0)
    )

    # 5. approval_ratio — approved_amount / billed_amount
    df['approval_ratio'] = df['approved_amount'] / df['billed_amount']
    pending_median = df.loc[df['claim_status'] == 'Pending', 'approval_ratio'].median()
    pending_median = pending_median if not pd.isna(pending_median) else 0.5
    df.loc[(df['claim_status'] == 'Rejected') & df['approval_ratio'].isna(), 'approval_ratio'] = 0.0
    df.loc[(df['claim_status'] == 'Pending')  & df['approval_ratio'].isna(), 'approval_ratio'] = pending_median
    df.loc[(df['claim_status'] == 'Paid')     & df['approval_ratio'].isna(), 'approval_ratio'] = 1.0

    # 6-8. Outlier-capped numeric columns
    df['billed_amount_capped'] = iqr_cap(df['billed_amount'])
    df['payment_days_capped']  = iqr_cap(df['payment_days']).fillna(0)
    df['los_capped']           = iqr_cap(df['length_of_stay_hours'])

    # 9-11. Time-based features
    df['visit_month']     = df['visit_date'].dt.month
    df['visit_dayofweek'] = df['visit_date'].dt.dayofweek
    df['visit_quarter']   = df['visit_date'].dt.quarter

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 13. CATEGORICAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply fixed label encoding using ENCODING_MAPS.
    Adds a new '<col>_encoded' column for each categorical column.

    Parameters
    ----------
    df : pd.DataFrame  DataFrame after engineer_features()

    Returns
    -------
    pd.DataFrame  Same DataFrame with _encoded suffix columns added
    """
    df = df.copy()
    for col, mapping in ENCODING_MAPS.items():
        if col in df.columns:
            df[col + '_encoded'] = df[col].map(mapping)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 14. FINAL MODEL TABLE BUILDER
# ══════════════════════════════════════════════════════════════════════════════
# Columns used by Phase 3 for Model A and Model B
MODEL_A_FEATURES = [
    'age', 'chronic_flag', 'los_capped', 'visit_frequency',
    'avg_los_per_patient', 'days_since_registration',
    'billed_amount_capped', 'approval_ratio', 'provider_rejection_rate',
    'payment_days_capped', 'visit_month', 'visit_dayofweek', 'visit_quarter',
    'department_encoded', 'visit_type_encoded', 'gender_encoded',
    'city_encoded', 'insurance_provider_encoded',
]

MODEL_B_FEATURES = MODEL_A_FEATURES  # same feature set for both models

MODEL_A_TARGET   = 'risk_score_encoded'
MODEL_B_TARGET   = 'claim_status_encoded'


def build_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering + encoding + column selection.
    Returns the final model-ready DataFrame.

    Parameters
    ----------
    df : pd.DataFrame  Output of get_merged_df()

    Returns
    -------
    pd.DataFrame  25,000 × 30 model table, zero nulls
    """
    df = engineer_features(df)
    df = encode_categoricals(df)

    model_cols = [
        'visit_id', 'patient_id', 'visit_date',
        'age', 'chronic_flag', 'los_capped', 'visit_frequency',
        'avg_los_per_patient', 'days_since_registration',
        'billed_amount_capped', 'approval_ratio',
        'provider_rejection_rate', 'payment_days_capped',
        'visit_month', 'visit_dayofweek', 'visit_quarter',
        'department', 'visit_type', 'gender', 'city', 'insurance_provider',
        'department_encoded', 'visit_type_encoded', 'gender_encoded',
        'city_encoded', 'insurance_provider_encoded',
        'risk_score',   'risk_score_encoded',
        'claim_status', 'claim_status_encoded',
    ]
    available = [c for c in model_cols if c in df.columns]
    return df[available]


# ══════════════════════════════════════════════════════════════════════════════
# 15. ONE-LINE NOTEBOOK SETUP
# ══════════════════════════════════════════════════════════════════════════════
def notebook_setup(
    phase_output_dir:  str  = None,
    patients_path:     str  = 'patients.csv',
    visits_path:       str  = 'visits.csv',
    billing_path:      str  = 'billing.csv',
    load_db:           bool = True,
    load_merged:       bool = True,
    verbose:           bool = True,
):
    """
    Single call that sets up everything a notebook needs.
    Call this at the top of any phase notebook.

    Parameters
    ----------
    phase_output_dir : str   Output folder name e.g. 'Output_Phase3'
                             Pass None to skip output folder creation.
    patients_path    : str   Path to patients.csv
    visits_path      : str   Path to visits.csv
    billing_path     : str   Path to billing.csv
    load_db          : bool  Build SQLite database (default True)
    load_merged      : bool  Return merged DataFrame (default True)
    verbose          : bool  Print progress (default True)

    Returns
    -------
    dict with keys:
        'con'      : sqlite3.Connection  (if load_db=True)
        'patients' : pd.DataFrame
        'visits'   : pd.DataFrame
        'billing'  : pd.DataFrame
        'df'       : pd.DataFrame merged  (if load_merged=True)
        'out'      : callable path helper (if phase_output_dir given)

    Usage — ONE line in any notebook
    ---------------------------------
    from phase1_utils import notebook_setup
    ctx = notebook_setup('Output_Phase3')

    con      = ctx['con']       # SQLite database
    df       = ctx['df']        # merged DataFrame
    out      = ctx['out']       # path helper: out('chart.png')
    patients = ctx['patients']  # raw patients table
    visits   = ctx['visits']    # raw visits table
    billing  = ctx['billing']   # raw billing table
    """
    configure_pandas()
    configure_plots()

    if verbose:
        print("🔧 phase1_utils — notebook setup")
        print("─" * 40)

    result = {}

    # ── Load database ──────────────────────────────────────────────────────
    if load_db:
        if verbose: print("📦 Loading CSVs into SQLite...")
        con, patients, visits, billing = setup_database(
            patients_path, visits_path, billing_path, verbose=verbose
        )
        result['con']      = con
        result['patients'] = patients
        result['visits']   = visits
        result['billing']  = billing

    # ── Merged DataFrame ───────────────────────────────────────────────────
    if load_merged:
        if verbose: print("🔗 Building merged DataFrame...")
        df = get_merged_df(
            result['patients'], result['visits'], result['billing'],
            verbose=verbose
        )
        result['df'] = df

    # ── Output directory ───────────────────────────────────────────────────
    if phase_output_dir:
        if verbose: print(f"📁 Creating output folder...")
        out = set_output_dir(phase_output_dir, verbose=verbose)
        result['out'] = out

    if verbose:
        print("─" * 40)
        print("✅ Setup complete. Everything is ready.\n")

    return result


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST
# Run this file directly to confirm everything works:
#   python phase1_utils.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Running phase1_utils self-test...\n")

    ctx = notebook_setup(
        phase_output_dir = 'Output_SelfTest',
        patients_path    = 'patients.csv',
        visits_path      = 'visits.csv',
        billing_path     = 'billing.csv',
    )

    df  = ctx['df']
    out = ctx['out']
    con = ctx['con']

    # Test feature engineering
    df_feat = engineer_features(df)
    df_enc  = encode_categoricals(df_feat)
    model   = build_model_table(df)

    print(f"Model table shape : {model.shape}")
    print(f"Null count        : {model.isnull().sum().sum()}")

    # Test run_query
    result = run_query(con, "Quick check",
                       "SELECT department, COUNT(*) AS visits FROM visits GROUP BY 1 LIMIT 3")

    import shutil
    shutil.rmtree('Output_SelfTest', ignore_errors=True)
    print("\n✅ Self-test passed — phase1_utils.py is working correctly.")
