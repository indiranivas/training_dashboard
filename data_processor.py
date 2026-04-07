import os
import pandas as pd
from genai_helper import build_categorical_index, reset_categorical_index

_DATA_CACHE = {"mtime": None, "df": None, "source": None}

def load_data():
    base_dir = os.path.dirname(__file__)
    xlsx_path = os.path.join(base_dir, "data.xlsx")
    csv_path = os.path.join(base_dir, "data.csv")

    xlsx_mtime = None
    csv_mtime = None
    try:
        xlsx_mtime = os.path.getmtime(xlsx_path)
    except OSError:
        pass
    try:
        csv_mtime = os.path.getmtime(csv_path)
    except OSError:
        pass

    if _DATA_CACHE["df"] is not None and _DATA_CACHE.get("source") == "xlsx" and _DATA_CACHE.get("mtime") == xlsx_mtime:
        return _DATA_CACHE["df"]
    if _DATA_CACHE["df"] is not None and _DATA_CACHE.get("source") == "csv" and _DATA_CACHE.get("mtime") == csv_mtime:
        return _DATA_CACHE["df"]

    df = None
    loaded_source = None
    mtime = None

    try:
        if xlsx_mtime is None:
            raise FileNotFoundError(xlsx_path)
        df = pd.read_excel(xlsx_path)
        loaded_source = "xlsx"
        mtime = xlsx_mtime
    except PermissionError:
        if csv_mtime is None:
            raise
        df = pd.read_csv(csv_path)
        loaded_source = "csv"
        mtime = csv_mtime
    except ValueError:
        if csv_mtime is None:
            raise
        df = pd.read_csv(csv_path)
        loaded_source = "csv"
        mtime = csv_mtime

    if df is None or loaded_source is None:
        raise FileNotFoundError(f"Missing dataset file: {xlsx_path} or {csv_path}")
    df.columns = [col.strip() for col in df.columns]

    if 'Department' in df.columns:
        df['Department'] = df['Department'].fillna('').astype(str).str.strip()
    if 'Business Unit' in df.columns:
        df['Business Unit'] = df['Business Unit'].fillna('').astype(str).str.strip()

    if 'Overall Training Duration (Planned Hrs)' in df.columns:
        df['Overall Training Duration (Planned Hrs)'] = pd.to_numeric(
            df['Overall Training Duration (Planned Hrs)'], errors='coerce'
        )
        emp_totals = df.groupby('Emp ID')['Overall Training Duration (Planned Hrs)'].transform('sum')
        df['Completed_20hrs'] = emp_totals >= 20

    reset_categorical_index()
    build_categorical_index(df)

    _DATA_CACHE["source"] = loaded_source
    _DATA_CACHE["mtime"] = mtime
    _DATA_CACHE["df"] = df
    return _DATA_CACHE["df"]


def categorize_hours(hours):
    if pd.isna(hours):
        return 'F'
    if hours >= 20:
        return 'A'
    elif hours >= 15:
        return 'B'
    elif hours >= 10:
        return 'C'
    elif hours >= 5:
        return 'D'
    elif hours > 0:
        return 'E'
    else:
        return 'F'


def categorize_completion_percentage(percentage):
    if pd.isna(percentage):
        return 'F'
    if percentage >= 100:
        return 'A'
    elif percentage >= 75:
        return 'B'
    elif percentage >= 50:
        return 'C'
    elif percentage >= 25:
        return 'D'
    elif percentage >= 1:
        return 'E'
    else:
        return 'F'


def compute_metrics(df):
    if df.empty:
        return {
            "coverage": 0, "avg_hours": 0, "completion_rate": 0, "total_emp": 0,
            "bu": pd.Series(dtype=float), "dept": pd.Series(dtype=float), "ttype": pd.Series(dtype=float),
            "dept_hours_categories": {}, "bu_hours_categories": {},
            "dept_completion_categories": {}, "bu_completion_categories": {}
        }

    emp_agg = df.groupby('Emp ID').agg(
        Total_Hours=('Overall Training Duration (Planned Hrs)', 'sum'),
        BU=('Business Unit', 'first'),
        Dept=('Department', 'first')
    )
    emp_agg['Completed'] = emp_agg['Total_Hours'] >= 20

    total_emp = len(emp_agg)
    completed_emp = emp_agg['Completed'].sum()

    completed_any_training = df[df['Training Status'].str.strip().str.lower() == 'completed']['Emp ID'].nunique() if 'Training Status' in df.columns else total_emp
    coverage = round((completed_any_training / total_emp) * 100, 1) if total_emp > 0 else 0
    
    completion_rate = round((completed_emp / total_emp) * 100, 1) if total_emp > 0 else 0

    unique_sessions = df.drop_duplicates(subset=['Training Name', 'Start Date']) if 'Training Name' in df.columns and 'Start Date' in df.columns else pd.DataFrame()
    if not unique_sessions.empty and 'Overall Training Duration (Planned Hrs)' in unique_sessions.columns:
        avg_hours = round(unique_sessions['Overall Training Duration (Planned Hrs)'].sum(), 1)
    else:
        avg_hours = 0

    bu = emp_agg.groupby('BU')['Completed'].mean() * 100
    dept = emp_agg.groupby('Dept')['Completed'].mean() * 100
    ttype = (df['Training Type'].value_counts(normalize=True) * 100).round(1) if 'Training Type' in df.columns else pd.Series(dtype=float)

    emp_agg['Hours_Category'] = emp_agg['Total_Hours'].apply(categorize_hours)
    
    dept_hours_dist = emp_agg.groupby(['Dept', 'Hours_Category']).size().unstack(fill_value=0)
    dept_hours_categories = {}
    for dept_name in dept_hours_dist.index:
        dept_total = dept_hours_dist.loc[dept_name].sum()
        dept_hours_categories[str(dept_name)] = {
            'A': int(dept_hours_dist.loc[dept_name].get('A', 0)),
            'B': int(dept_hours_dist.loc[dept_name].get('B', 0)),
            'C': int(dept_hours_dist.loc[dept_name].get('C', 0)),
            'D': int(dept_hours_dist.loc[dept_name].get('D', 0)),
            'E': int(dept_hours_dist.loc[dept_name].get('E', 0)),
            'F': int(dept_hours_dist.loc[dept_name].get('F', 0)),
            'total': dept_total
        }

    bu_hours_dist = emp_agg.groupby(['BU', 'Hours_Category']).size().unstack(fill_value=0)
    bu_hours_categories = {}
    for bu_name in bu_hours_dist.index:
        bu_total = bu_hours_dist.loc[bu_name].sum()
        bu_hours_categories[str(bu_name)] = {
            'A': int(bu_hours_dist.loc[bu_name].get('A', 0)),
            'B': int(bu_hours_dist.loc[bu_name].get('B', 0)),
            'C': int(bu_hours_dist.loc[bu_name].get('C', 0)),
            'D': int(bu_hours_dist.loc[bu_name].get('D', 0)),
            'E': int(bu_hours_dist.loc[bu_name].get('E', 0)),
            'F': int(bu_hours_dist.loc[bu_name].get('F', 0)),
            'total': bu_total
        }

    dept_completion_pct = dept
    dept_completion_categories = {}
    for dept_name in dept_completion_pct.index:
        pct = dept_completion_pct.loc[dept_name]
        dept_completion_categories[str(dept_name)] = {
            'completion_pct': round(pct, 1),
            'category': categorize_completion_percentage(pct)
        }

    bu_completion_pct = bu
    bu_completion_categories = {}
    for bu_name in bu_completion_pct.index:
        pct = bu_completion_pct.loc[bu_name]
        bu_completion_categories[str(bu_name)] = {
            'completion_pct': round(pct, 1),
            'category': categorize_completion_percentage(pct)
        }

    return {
        "coverage": coverage,
        "avg_hours": avg_hours,
        "completion_rate": completion_rate,
        "total_emp": total_emp,
        "bu": bu,
        "dept": dept,
        "ttype": ttype,
        "dept_hours_categories": dept_hours_categories,
        "bu_hours_categories": bu_hours_categories,
        "dept_completion_categories": dept_completion_categories,
        "bu_completion_categories": bu_completion_categories
    }
