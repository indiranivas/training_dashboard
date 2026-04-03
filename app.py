from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from genai_helper import (
    generate_query_plan,
    execute_query_plan,
    generate_rag_response,
    build_categorical_index,
    reset_categorical_index,
)

app = Flask(__name__)

_DATA_CACHE = {"mtime": None, "df": None}

def load_data():
    path = "data.xlsx"
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        raise FileNotFoundError(f"Missing required dataset file: {path}")

    if _DATA_CACHE["df"] is not None and _DATA_CACHE["mtime"] == mtime:
        return _DATA_CACHE["df"]

    df = pd.read_excel(path)
    df.columns = [col.strip() for col in df.columns]

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

    _DATA_CACHE["mtime"] = mtime
    _DATA_CACHE["df"] = df
    return _DATA_CACHE["df"]


def compute_metrics(df):
    if df.empty:
        return {
            "coverage": 0, "avg_hours": 0, "completion_rate": 0, "total_emp": 0,
            "bu": pd.Series(dtype=float), "dept": pd.Series(dtype=float),
            "ttype": pd.Series(dtype=float)
        }

    emp_agg = df.groupby('Emp ID').agg(
        Total_Hours=('Overall Training Duration (Planned Hrs)', 'sum'),
        BU=('Business Unit', 'first'),
        Dept=('Department', 'first')
    )
    emp_agg['Completed'] = emp_agg['Total_Hours'] >= 20

    total_emp     = len(emp_agg)
    completed_emp = emp_agg['Completed'].sum()

    coverage        = round((completed_emp / total_emp) * 100, 1) if total_emp > 0 else 0
    completion_rate = coverage

    unique_sessions = df.drop_duplicates(subset=['Training Name', 'Start Date'])
    avg_hours       = round(unique_sessions['Overall Training Duration (Planned Hrs)'].sum(), 1)

    bu    = emp_agg.groupby('BU')['Completed'].mean() * 100
    dept  = emp_agg.groupby('Dept')['Completed'].mean() * 100
    ttype = (df['Training Type'].value_counts(normalize=True) * 100).round(1)

    return {
        "coverage": coverage,
        "avg_hours": avg_hours,
        "completion_rate": completion_rate,
        "total_emp": total_emp,
        "bu": bu,
        "dept": dept,
        "ttype": ttype,
    }


@app.route("/")
def dashboard():
    df = load_data()

    departments    = sorted([str(d) for d in df['Department'].dropna().unique()])
    business_units = sorted([str(b) for b in df['Business Unit'].dropna().unique()])

    dept_filter = request.args.get('department', 'All')
    bu_filter   = request.args.get('business_unit', 'All')

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
        type_values=list(metrics["ttype"].round(1).values),
    )


@app.route("/employees")
def employees():
    df = load_data()

    departments    = sorted([str(d) for d in df['Department'].dropna().unique()])
    business_units = sorted([str(b) for b in df['Business Unit'].dropna().unique()])

    dept_filter = request.args.get('department', 'All')
    bu_filter   = request.args.get('business_unit', 'All')

    if dept_filter != 'All':
        df = df[df['Department'] == dept_filter]
    if bu_filter != 'All':
        df = df[df['Business Unit'] == bu_filter]

    agg_df = df.groupby(['Emp ID', 'Employee Name', 'Department', 'Business Unit']).agg(
        Total_Hours=('Overall Training Duration (Planned Hrs)', 'sum'),
        Training_Count=('Training Name', 'nunique')
    ).reset_index()

    agg_df['Completed_20hrs'] = agg_df['Total_Hours'] >= 20
    agg_df = agg_df.sort_values(by='Total_Hours', ascending=False)

    try:
        page = int(request.args.get('page', 1))
    except ValueError:
        page = 1

    per_page      = 50
    total_records = len(agg_df)
    total_pages   = (total_records + per_page - 1) // per_page

    start_idx = (page - 1) * per_page
    end_idx   = start_idx + per_page
    records   = agg_df.iloc[start_idx:end_idx].fillna("").to_dict('records')

    all_emp_count = load_data()['Emp ID'].nunique()

    return render_template(
        "Employees.html",
        records=records,
        departments=departments,
        business_units=business_units,
        dept_filter=dept_filter,
        bu_filter=bu_filter,
        filtered_count=total_records,
        all_count=all_emp_count,
        page=page,
        total_pages=total_pages,
    )


@app.route("/analytics")
def analytics():
    df = load_data()

    start_date = request.args.get('start_date')
    end_date   = request.args.get('end_date')

    if start_date:
        df = df[pd.to_datetime(df['Start Date'], errors='coerce') >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df['End Date'], errors='coerce') <= pd.to_datetime(end_date)]

    # Monthly Trend
    if 'Training Start Month' in df.columns:
        # FIX: cert rate per month should be based on unique employees, not row counts
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
        months          = list(trend_df['Training Start Month'].astype(str))
        trend_total     = [int(x) for x in trend_df['total'].values]
        trend_certified = [int(x) for x in trend_df['certified'].values]
    else:
        months, trend_total, trend_certified = [], [], []

    # Dept Cert Rates — FIX: use employee-level aggregation, not row counts
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
    dept_stats  = dept_stats.sort_values(by='cert_rate', ascending=False)
    dept_labels = list(dept_stats['Department'].astype(str))
    dept_rates  = [float(x) for x in dept_stats['cert_rate'].values]

    return render_template(
        "Analytics.html",
        months=months,
        trend_total=trend_total,
        trend_certified=trend_certified,
        dept_labels=dept_labels,
        dept_rates=dept_rates,
        total_records=len(df),
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
            "query":          message,
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
        "query":          message,
        "gemini_plan":    gemini_plan,
        "data_result":    data_result,
        "final_response": final_response,
    })


if __name__ == "__main__":
    app.run(debug=True)
