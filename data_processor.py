import os
import re
import json
import pandas as pd
from genai_helper import build_categorical_index, reset_categorical_index

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "data_config.json")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data_files")
_DATA_CACHE = {}

def get_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                
                # Migrate older schema to new schema dynamically if needed
                if "active_year" in data and "years" not in data:
                    migrated = {"years": {}}
                    act_yr = data.get("active_year")
                    if act_yr:
                        migrated["years"][act_yr] = {"link": None, "active": 1}
                    links = data.get("live_links", {})
                    for y, l in links.items():
                        if y not in migrated["years"]:
                            migrated["years"][y] = {"link": l, "active": 0}
                        else:
                            migrated["years"][y]["link"] = l
                    return migrated
                return data
        except Exception:
            pass
    return {"years": {}}

def save_config(config_data):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config_data, f, indent=4)

def get_active_year():
    config = get_config()
    for y, info in config.get("years", {}).items():
        if info.get("active") == 1:
            return str(y)
    return None

def set_active_year(year):
    config = get_config()
    if "years" not in config: config["years"] = {}
    if str(year) not in config["years"]:
        config["years"][str(year)] = {"link": None, "active": 1}
        
    for y in config["years"]:
        config["years"][y]["active"] = 1 if str(y) == str(year) else 0
    save_config(config)

def get_live_link(year):
    return get_config().get("years", {}).get(str(year), {}).get("link")

def set_live_link(year, link):
    config = get_config()
    if "years" not in config: config["years"] = {}
    if str(year) not in config["years"]:
        config["years"][str(year)] = {"link": None, "active": 0}
    config["years"][str(year)]["link"] = link.strip()
    save_config(config)

def list_available_years():
    if not os.path.exists(DATA_DIR):
        return []

    years = set()
    for filename in os.listdir(DATA_DIR):
        match = re.match(r"^data_(\d{4})\.(csv|xlsx|xls)$", filename, re.IGNORECASE)
        if match:
            years.add(match.group(1))
    return sorted(years, reverse=True)

def _resolve_dataset_paths(year=None):
    if year:
        return (
            os.path.join(DATA_DIR, f"data_{year}.xlsx"),
            os.path.join(DATA_DIR, f"data_{year}.csv"),
        )
    return (
        os.path.join(DATA_DIR, "data.xlsx"),
        os.path.join(DATA_DIR, "data.csv"),
    )

def _normalize_dataframe(df, build_index=False):
    if df is None:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    for col in ['Department', 'Business Unit', 'Training Type', 'Training Status', 'Employee Status', 'Quarter', 'Designation', 'Training Source', 'Trainer Name', 'Training Name', 'Employee Name']:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()

    if 'Overall Training Duration (Planned Hrs)' in df.columns:
        df['Overall Training Duration (Planned Hrs)'] = pd.to_numeric(
            df['Overall Training Duration (Planned Hrs)'], errors='coerce'
        )

    if 'Emp ID' in df.columns and 'Overall Training Duration (Planned Hrs)' in df.columns:
        emp_totals = df.groupby('Emp ID')['Overall Training Duration (Planned Hrs)'].transform('sum')
        df['Completed_20hrs'] = emp_totals >= 20

    if build_index:
        reset_categorical_index()
        build_categorical_index(df)

    return df

def _read_dataset(xlsx_path, csv_path):
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

    df = None
    loaded_source = None
    mtime = None

    try:
        if xlsx_mtime is None:
            raise FileNotFoundError(xlsx_path)
        df = pd.read_excel(xlsx_path)
        loaded_source = "xlsx"
        mtime = xlsx_mtime
    except FileNotFoundError:
        if csv_mtime is not None:
            df = pd.read_csv(csv_path)
            loaded_source = "csv"
            mtime = csv_mtime
    except (PermissionError, ValueError):
        if csv_mtime is not None:
            df = pd.read_csv(csv_path)
            loaded_source = "csv"
            mtime = csv_mtime

    return df, loaded_source, mtime, xlsx_mtime, csv_mtime

def _get_file_mtimes(xlsx_path, csv_path):
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
    return xlsx_mtime, csv_mtime

def load_data_for_year(year, build_index=False):
    year = str(year).strip()
    cache_key = f"year:{year}"
    xlsx_path, csv_path = _resolve_dataset_paths(year)
    xlsx_mtime, csv_mtime = _get_file_mtimes(xlsx_path, csv_path)

    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        if cached.get("source") == "xlsx" and cached.get("mtime") == xlsx_mtime and xlsx_mtime is not None:
            if build_index:
                reset_categorical_index()
                build_categorical_index(cached["df"])
            return cached["df"]
        if cached.get("source") == "csv" and cached.get("mtime") == csv_mtime and csv_mtime is not None:
            if build_index:
                reset_categorical_index()
                build_categorical_index(cached["df"])
            return cached["df"]

    df, loaded_source, mtime, xlsx_mtime, csv_mtime = _read_dataset(xlsx_path, csv_path)
    if df is None or loaded_source is None:
        return pd.DataFrame()

    normalized_df = _normalize_dataframe(df, build_index=build_index)
    _DATA_CACHE[cache_key] = {"source": loaded_source, "mtime": mtime, "df": normalized_df}
    return normalized_df

def load_data():
    active_year = get_active_year()
    if active_year:
        return load_data_for_year(active_year, build_index=True)

    cache_key = "default"
    xlsx_path, csv_path = _resolve_dataset_paths()
    xlsx_mtime, csv_mtime = _get_file_mtimes(xlsx_path, csv_path)

    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        if cached.get("source") == "xlsx" and cached.get("mtime") == xlsx_mtime and xlsx_mtime is not None:
            reset_categorical_index()
            build_categorical_index(cached["df"])
            return cached["df"]
        if cached.get("source") == "csv" and cached.get("mtime") == csv_mtime and csv_mtime is not None:
            reset_categorical_index()
            build_categorical_index(cached["df"])
            return cached["df"]

    df, loaded_source, mtime, xlsx_mtime, csv_mtime = _read_dataset(xlsx_path, csv_path)
    if df is None or loaded_source is None:
        return pd.DataFrame()

    normalized_df = _normalize_dataframe(df, build_index=True)
    _DATA_CACHE[cache_key] = {"source": loaded_source, "mtime": mtime, "df": normalized_df}
    return normalized_df


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
            "ttype_20hrs": pd.Series(dtype=float),
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
    if 'Training Type' in df.columns and 'Overall Training Duration (Planned Hrs)' in df.columns:
        ttype_20hrs = (
            df[df['Overall Training Duration (Planned Hrs)'] >= 20]
            ['Training Type']
            .value_counts(normalize=True) * 100
        ).round(1)
    else:
        ttype_20hrs = pd.Series(dtype=float)

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
        "ttype_20hrs": ttype_20hrs,
        "dept_hours_categories": dept_hours_categories,
        "bu_hours_categories": bu_hours_categories,
        "dept_completion_categories": dept_completion_categories,
        "bu_completion_categories": bu_completion_categories
    }
