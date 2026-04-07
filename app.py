from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from urllib.parse import quote_plus
import numpy as np
from genai_helper import (
    generate_query_plan,
    execute_query_plan,
    generate_rag_response,
    build_categorical_index,
    reset_categorical_index,
)

app = Flask(__name__)

# URL encoder for dashboards to handle special chars like ampersand safely
app.jinja_env.filters['urlencode'] = lambda value: quote_plus(str(value))

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

    # Prefer Excel, but if the XLSX is locked by OneDrive/Excel, fall back to CSV.
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
        # Sometimes Excel engines fail to parse; CSV is a safe fallback.
        if csv_mtime is None:
            raise
        df = pd.read_csv(csv_path)
        loaded_source = "csv"
        mtime = csv_mtime

    if df is None or loaded_source is None:
        raise FileNotFoundError(f"Missing dataset file: {xlsx_path} or {csv_path}")
    df.columns = [col.strip() for col in df.columns]

    # Trim whitespace and fix values in key columns to avoid URL filter mismatch
    if 'Department' in df.columns:
        df['Department'] = df['Department'].fillna('').astype(str).str.strip()
    if 'Business Unit' in df.columns:
        df['Business Unit'] = df['Business Unit'].fillna('').astype(str).str.strip()

    df['Overall Training Duration (Planned Hrs)'] = pd.to_numeric(
        df['Overall Training Duration (Planned Hrs)'], errors='coerce'
    )

    # FIX: Completed_20hrs must reflect each employee's TOTAL hours, not per-row hours.
    # A row-level flag >= 20 is meaningless for multi-session employees.
    emp_totals = df.groupby('Emp ID')['Overall Training Duration (Planned Hrs)'].transform('sum')
    df['Completed_20hrs'] = emp_totals >= 20

    # Rebuild semantic dictionary index when data changes
    reset_categorical_index()
    build_categorical_index(df)

    _DATA_CACHE["source"] = loaded_source
    _DATA_CACHE["mtime"] = mtime
    _DATA_CACHE["df"] = df
    return _DATA_CACHE["df"]


def categorize_hours(hours):
    """Categorize training hours into A-F categories"""
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
    """Categorize completion percentage into A-F categories"""
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

    # Employee-level Aggregation for Accurate Completion Rate
    emp_agg = df.groupby('Emp ID').agg(
        Total_Hours=('Overall Training Duration (Planned Hrs)', 'sum'),
        BU=('Business Unit', 'first'),
        Dept=('Department', 'first')
    )
    emp_agg['Completed'] = emp_agg['Total_Hours'] >= 20

    total_emp = len(emp_agg)
    completed_emp = emp_agg['Completed'].sum()

    # Coverage: % of employees who have completed at least one training
    completed_any_training = df[df['Training Status'].str.strip().str.lower() == 'completed']['Emp ID'].nunique() if 'Training Status' in df.columns else total_emp
    coverage = round((completed_any_training / total_emp) * 100, 1) if total_emp > 0 else 0
    
    # Completion Rate: % of employees who achieved the 20+ hours target
    completion_rate = round((completed_emp / total_emp) * 100, 1) if total_emp > 0 else 0

    unique_sessions = df.drop_duplicates(subset=['Training Name', 'Start Date'])
    avg_hours = round(unique_sessions['Overall Training Duration (Planned Hrs)'].sum(), 1)

    bu = emp_agg.groupby('BU')['Completed'].mean() * 100
    dept = emp_agg.groupby('Dept')['Completed'].mean() * 100
    ttype = (df['Training Type'].value_counts(normalize=True) * 100).round(1)

    # =============== DEPARTMENT-WISE HOURS CATEGORIES ===============
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

    # =============== BU-WISE HOURS CATEGORIES ===============
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

    # =============== DEPARTMENT-WISE COMPLETION PERCENTAGE CATEGORIES ===============
    dept_completion_pct = dept
    dept_completion_categories = {}
    for dept_name in dept_completion_pct.index:
        pct = dept_completion_pct.loc[dept_name]
        dept_completion_categories[str(dept_name)] = {
            'completion_pct': round(pct, 1),
            'category': categorize_completion_percentage(pct)
        }

    # =============== BU-WISE COMPLETION PERCENTAGE CATEGORIES ===============
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


@app.route("/")
def dashboard():
    df = load_data()

    departments = sorted([str(d) for d in df['Department'].dropna().unique()])
    business_units = sorted([str(b) for b in df['Business Unit'].dropna().unique()])

    dept_filter = request.args.get('department', 'All')
    bu_filter = request.args.get('business_unit', 'All')

    if dept_filter != 'All':
        df = df[df['Department'] == dept_filter]

    if bu_filter != 'All':
        df = df[df['Business Unit'] == bu_filter]

    metrics = compute_metrics(df)
    records = df.head(50).fillna("").to_dict('records')

    return render_template(
        "Dashboard.html",
        metrics=metrics,
        records=records,
        departments=departments,
        business_units=business_units,
        dept_filter=dept_filter,
        bu_filter=bu_filter,
        bu_labels=list(metrics["bu"].index.astype(str)),
        bu_values=list(metrics["bu"].round(1).values),
        dept_labels=list(metrics["dept"].index.astype(str)),
        dept_values=list(metrics["dept"].round(1).values),
        type_labels=list(metrics["ttype"].index.astype(str)),
        type_values=list(metrics["ttype"].round(1).values)
    )


@app.route("/employees")
def employees():
    df = load_data()
    departments = sorted([str(d) for d in df['Department'].dropna().unique()])
    business_units = sorted([str(b) for b in df['Business Unit'].dropna().unique()])
    dept_filter = request.args.get('department', 'All')
    bu_filter = request.args.get('business_unit', 'All')
    hours_category = request.args.get('hours_category', 'All')
    status_filter = request.args.get('employee_status', 'All')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Apply Custom Date Range filtering logic matching Analytics
    if start_date:
        df = df[pd.to_datetime(df['Start Date'], errors='coerce') >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df['End Date'], errors='coerce') <= pd.to_datetime(end_date)]

    # Aggregate by employee to guarantee 1:1 mapping with analytics calculations
    agg_df = df.groupby('Emp ID').agg(**{
        'Employee Name': ('Employee Name', 'first'),
        'Department': ('Department', 'first'),
        'Business Unit': ('Business Unit', 'first'),
        'Employee_Status': ('Employee Status', 'first'),
        'Total_Hours': ('Overall Training Duration (Planned Hrs)', 'sum'),
        'Training_Count': ('Training Name', 'nunique'),
        'Course_List': ('Training Name', lambda x: ', '.join(sorted(x.dropna().astype(str).unique())))
    }).reset_index()

    # Re-calculate health based on total aggregated hours
    agg_df['Completed_20hrs'] = agg_df['Total_Hours'] >= 20
    agg_df['Hours_Category'] = agg_df['Total_Hours'].apply(categorize_hours)

    # NOW apply filters matching the UI selection
    if dept_filter != 'All':
        agg_df = agg_df[agg_df['Department'] == dept_filter]
    if bu_filter != 'All':
        agg_df = agg_df[agg_df['Business Unit'] == bu_filter]
    if hours_category and hours_category != 'All':
        agg_df = agg_df[agg_df['Hours_Category'] == hours_category]
    if status_filter != 'All':
        agg_df = agg_df[agg_df['Employee_Status'] == status_filter]

    agg_df = agg_df.sort_values(by='Total_Hours', ascending=False)

    try:
        page = int(request.args.get('page', 1))
    except ValueError:
        page = 1

    per_page = 50
    total_records = len(agg_df)
    total_pages = (total_records + per_page - 1) // per_page

    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page

    records = agg_df.iloc[start_idx:end_idx].fillna("").to_dict('records')

    # Total employees overall globally
    all_emp_count = load_data()['Emp ID'].nunique()

    # Build course details per employee for modal
    employee_courses = {}
    detail_columns = ['Training Name', 'Overall Training Duration (Planned Hrs)', 'Training Status', 'Training Source', 'Start Date', 'End Date']
    if all(col in df.columns for col in detail_columns):
        for emp_id, group in df.groupby('Emp ID'):
            employee_courses[str(emp_id)] = group[detail_columns].fillna('').to_dict('records')
    else:
        for emp_id, group in df.groupby('Emp ID'):
            employee_courses[str(emp_id)] = group.to_dict('records')

    return render_template(
        "Employees.html",
        records=records,
        departments=departments,
        business_units=business_units,
        employee_statuses=sorted([str(s) for s in df['Employee Status'].dropna().unique()]),
        dept_filter=dept_filter,
        bu_filter=bu_filter,
        status_filter=status_filter,
        hours_category=hours_category,
        employee_courses=employee_courses,
        filtered_count=total_records,
        all_count=all_emp_count,
        page=page,
        total_pages=total_pages
    )


@app.route("/analytics")
def analytics():
    df = load_data()

    # Filters
    dept_filter = request.args.get('department', 'All')
    bu_filter = request.args.get('business_unit', 'All')

    # Custom Date Range filtering logic
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    if start_date:
        df = df[pd.to_datetime(df['Start Date'], errors='coerce') >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df['End Date'], errors='coerce') <= pd.to_datetime(end_date)]

    if dept_filter != 'All':
        df = df[df['Department'] == dept_filter]
    if bu_filter != 'All':
        df = df[df['Business Unit'] == bu_filter]

    # 1. Monthly Trend - FIX: cert rate per month should be based on unique employees, not row counts
    if 'Training Start Month' in df.columns:
        monthly_emp = (
            df.groupby(['Training Start Month', 'Emp ID'])['Overall Training Duration (Planned Hrs)']
            .sum()
            .reset_index()
        )
        monthly_emp['certified'] = monthly_emp['Overall Training Duration (Planned Hrs)'] >= 20
        trend_df = monthly_emp.groupby('Training Start Month').agg(
            total=('Emp ID', 'count'),
            certified=('certified', 'sum')
        ).reset_index()
        months = list(trend_df['Training Start Month'].astype(str))
        trend_total = [int(x) for x in trend_df['total'].values]
        trend_certified = [int(x) for x in trend_df['certified'].values]
    else:
        months, trend_total, trend_certified = [], [], []

    # 2. Dept Cert Rates — FIX: use employee-level aggregation, not row counts
    dept_emp = (
        df.groupby(['Department', 'Emp ID'])['Overall Training Duration (Planned Hrs)']
        .sum()
        .reset_index()
    )
    dept_emp['certified'] = dept_emp['Overall Training Duration (Planned Hrs)'] >= 20
    dept_stats = dept_emp.groupby('Department').agg(
        total=('Emp ID', 'count'),
        certified=('certified', 'sum')
    ).reset_index()
    dept_stats['cert_rate'] = (
        dept_stats['certified'] / dept_stats['total'] * 100
    ).fillna(0).round(1)
    dept_stats = dept_stats.sort_values(by='cert_rate', ascending=False)
    dept_labels = list(dept_stats['Department'].astype(str))
    dept_rates = [float(x) for x in dept_stats['cert_rate'].values]

    # 3. Compute full metrics for categories
    metrics = compute_metrics(df)

    return render_template(
        "Analytics.html",
        months=months,
        trend_total=trend_total,
        trend_certified=trend_certified,
        dept_labels=dept_labels,
        dept_rates=dept_rates,
        total_records=len(df),
        metrics=metrics
    )


@app.route("/settings")
def settings():
    df = load_data()
    return render_template(
        "settings.html",
        row_count=len(df),
        col_count=len(df.columns),
        columns=list(df.columns),
    )

@app.route("/api/insight", methods=["POST"])
def generate_insight():
    """
    Returns a short insight summary for the current page context.
    This is deterministic (no LLM) and respects filters provided by the UI.
    """
    body = request.get_json(silent=True) or {}
    context = body.get("context", {}) or {}

    page = str(context.get("page", "dashboard")).strip().lower()
    dept_filter = str(context.get("department", "All")).strip()
    bu_filter = str(context.get("business_unit", "All")).strip()
    start_date = context.get("start_date")
    end_date = context.get("end_date")

    df = load_data()

    # Apply common filters
    if dept_filter and dept_filter != "All" and "Department" in df.columns:
        df = df[df["Department"] == dept_filter]
    if bu_filter and bu_filter != "All" and "Business Unit" in df.columns:
        df = df[df["Business Unit"] == bu_filter]

    # Date filters (analytics)
    if (start_date or end_date) and "Start Date" in df.columns and "End Date" in df.columns:
        if start_date:
            df = df[pd.to_datetime(df["Start Date"], errors="coerce") >= pd.to_datetime(start_date)]
        if end_date:
            df = df[pd.to_datetime(df["End Date"], errors="coerce") <= pd.to_datetime(end_date)]

    if df.empty:
        return jsonify({
            "insight": "No data found for the selected filters. Try clearing filters or adjusting the date range.",
            "context_used": {"page": page, "department": dept_filter, "business_unit": bu_filter, "start_date": start_date, "end_date": end_date},
        })

    metrics = compute_metrics(df)

    # Helper: format top items
    def _top_series(s: pd.Series, n=3, ascending=False):
        if s is None or len(s) == 0:
            return []
        s2 = s.dropna()
        if len(s2) == 0:
            return []
        s2 = s2.sort_values(ascending=ascending).head(n)
        return [(str(idx), float(val)) for idx, val in s2.items()]

    # Dept/BU completion % (employee-level)
    top_depts = _top_series(metrics.get("dept"), n=3, ascending=False)
    low_depts = _top_series(metrics.get("dept"), n=1, ascending=True)
    top_bus = _top_series(metrics.get("bu"), n=3, ascending=False)
    low_bus = _top_series(metrics.get("bu"), n=1, ascending=True)

    # Most common training type
    ttype = metrics.get("ttype")
    top_type = None
    if isinstance(ttype, pd.Series) and len(ttype) > 0:
        top_type = (str(ttype.index[0]), float(ttype.iloc[0]))

    # Employee completion gap
    emp_agg = df.groupby("Emp ID").agg(
        Total_Hours=("Overall Training Duration (Planned Hrs)", "sum"),
        Employee_Name=("Employee Name", "first") if "Employee Name" in df.columns else ("Emp ID", "first"),
        Department=("Department", "first") if "Department" in df.columns else ("Emp ID", "first"),
        Business_Unit=("Business Unit", "first") if "Business Unit" in df.columns else ("Emp ID", "first"),
    )
    emp_agg["Completed"] = emp_agg["Total_Hours"] >= 20
    remaining = (20 - emp_agg["Total_Hours"]).clip(lower=0)
    # Nearest-to-complete (but not complete)
    near = (
        emp_agg[~emp_agg["Completed"]]
        .assign(Remaining=remaining[~emp_agg["Completed"]])
        .sort_values(by="Remaining", ascending=True)
        .head(5)
    )

    lines = []
    lines.append("**Insight summary (based on current filters):**")
    lines.append(f"- **Learning coverage (20+ hrs)**: **{metrics.get('coverage', 0)}%** across **{metrics.get('total_emp', 0)}** employees")
    lines.append(f"- **Total session hours (unique sessions)**: **{metrics.get('avg_hours', 0)}**")
    if top_type:
        lines.append(f"- **Most common training type**: **{top_type[0]}** ({top_type[1]}%)")

    if top_depts:
        lines.append("- **Top departments by completion %**:")
        for name, val in top_depts:
            lines.append(f"  - {name}: {val:.1f}%")
    if low_depts:
        lines.append(f"- **Lowest department**: **{low_depts[0][0]}** ({low_depts[0][1]:.1f}%)")

    if top_bus:
        lines.append("- **Top business units by completion %**:")
        for name, val in top_bus:
            lines.append(f"  - {name}: {val:.1f}%")
    if low_bus:
        lines.append(f"- **Lowest business unit**: **{low_bus[0][0]}** ({low_bus[0][1]:.1f}%)")

    if len(near) > 0:
        lines.append("- **Employees closest to 20 hours (actionable follow-up)**:")
        for _, row in near.iterrows():
            nm = str(row.get("Employee_Name", "")).strip()
            eid = str(_)
            rem = float(row.get("Remaining", 0))
            lines.append(f"  - {nm} (Emp ID {eid}): ~{rem:.1f} hrs remaining")

    # Extra: employee ranking for Employees page
    if page == "employees":
        top_emp = emp_agg.sort_values(by="Total_Hours", ascending=False).head(5)
        lines.append("- **Top employees by total hours**:")
        for eid, row in top_emp.iterrows():
            nm = str(row.get("Employee_Name", "")).strip()
            hrs = float(row.get("Total_Hours", 0))
            lines.append(f"  - {nm} (Emp ID {eid}): {hrs:.1f} hrs")

    # Extra: trend hint for Analytics page
    if page == "analytics" and "Training Start Month" in df.columns:
        monthly_emp = (
            df.groupby(["Training Start Month", "Emp ID"])["Overall Training Duration (Planned Hrs)"]
            .sum()
            .reset_index()
        )
        monthly_emp["certified"] = monthly_emp["Overall Training Duration (Planned Hrs)"] >= 20
        trend_df = monthly_emp.groupby("Training Start Month").agg(
            total=("Emp ID", "count"),
            certified=("certified", "sum")
        ).reset_index()
        if len(trend_df) >= 2:
            last = trend_df.iloc[-1]
            prev = trend_df.iloc[-2]
            delta = int(last["certified"]) - int(prev["certified"])
            direction = "up" if delta >= 0 else "down"
            lines.append(f"- **Latest month certified change**: **{direction} {abs(delta)}** vs previous month")

    return jsonify({
        "insight": "\n".join(lines),
        "context_used": {"page": page, "department": dept_filter, "business_unit": bu_filter, "start_date": start_date, "end_date": end_date},
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    FIXES:
    1. Schema passed to Gemini is now a flat {col: dtype} dict (not nested).
    2. execute_query_plan from genai_helper is used (has sort + limit).
    3. generate_rag_response is now actually called so the final answer
       is based on real data, not Gemini's pre-execution guess.
    4. execute_pandas_query (the broken duplicate) is deleted entirely.
    """
    body = request.get_json(silent=True) or {}
    message = body.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided."}), 400

    df = load_data()

    # FIX 1: flat schema — Gemini needs {col: dtype}, not {"columns":[], "dtypes":{}}
    schema_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # FIX 2: generate plan
    gemini_plan = generate_query_plan(message, schema_dict)

    if gemini_plan.get("intent") == "error":
        return jsonify({
            "query": message,
            "final_response": (
                gemini_plan.get("response_text")
                or "We encountered an issue interpreting your request. Please check your GEMINI_API_KEY."
            ),
            "data_result": [],
        }), 500

    # FIX 3: execute using the complete executor (with sort + limit)
    data_result = execute_query_plan(df, gemini_plan)

    # FIX 4: generate a real RAG answer from actual data — not the pre-execution guess
    final_response = generate_rag_response(message, data_result, gemini_plan)

    return jsonify({
        "query": message,
        "gemini_plan": gemini_plan,
        "data_result": data_result,
        "final_response": final_response,
    })


if __name__ == "__main__":
    app.run(debug=True)
