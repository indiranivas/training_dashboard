from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

def load_data():
    df = pd.read_excel("data.xlsx")
    df.columns = [col.strip() for col in df.columns]

    # Convert hours
    df['Overall Training Duration (Planned Hrs)'] = pd.to_numeric(
        df['Overall Training Duration (Planned Hrs)'], errors='coerce'
    )

    # Completion logic
    df['Completed_20hrs'] = df['Overall Training Duration (Planned Hrs)'] >= 20

    return df


def compute_metrics(df):
    if df.empty:
        return {
            "coverage": 0, "avg_hours": 0, "completion_rate": 0, "total_emp": 0,
            "bu": pd.Series(dtype=float), "dept": pd.Series(dtype=float), "ttype": pd.Series(dtype=float)
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

    coverage = round((completed_emp / total_emp) * 100, 1) if total_emp > 0 else 0
    completion_rate = coverage  # Completion rate securely maps to aggregated employees

    # Deduplicate sessions by Training Name and Start Date so we don't multiply durations by employee counts
    unique_sessions = df.drop_duplicates(subset=['Training Name', 'Start Date'])
    avg_hours = round(unique_sessions['Overall Training Duration (Planned Hrs)'].sum(), 1)
    
    # BU based on aggregated employee completion
    bu = emp_agg.groupby('BU')['Completed'].mean() * 100

    # Department based on aggregated employee completion
    dept = emp_agg.groupby('Dept')['Completed'].mean() * 100

    # Training Type (Distribution of types percentages across all training)
    ttype = (df['Training Type'].value_counts(normalize=True) * 100).round(1)

    return {
        "coverage": coverage,
        "avg_hours": avg_hours,
        "completion_rate": completion_rate,
        "total_emp": total_emp,
        "bu": bu,
        "dept": dept,
        "ttype": ttype
    }


@app.route("/")
def dashboard():
    df = load_data()
    
    # Get unique options before filtering
    departments = sorted([str(d) for d in df['Department'].dropna().unique()])
    business_units = sorted([str(b) for b in df['Business Unit'].dropna().unique()])

    # Get query params
    dept_filter = request.args.get('department', 'All')
    bu_filter = request.args.get('business_unit', 'All')

    # Apply filters
    if dept_filter != 'All':
        df = df[df['Department'] == dept_filter]
    
    if bu_filter != 'All':
        df = df[df['Business Unit'] == bu_filter]

    metrics = compute_metrics(df)
    
    # Get top 50 records for frontend table
    # Mapping NA to blank strings so JSON serialization doesn't fail with NaNs.
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

    if dept_filter != 'All':
        df = df[df['Department'] == dept_filter]
    if bu_filter != 'All':
        df = df[df['Business Unit'] == bu_filter]

    # Aggregate by employee to remove duplicates
    agg_df = df.groupby(['Emp ID', 'Employee Name', 'Department', 'Business Unit']).agg(
        Total_Hours=('Overall Training Duration (Planned Hrs)', 'sum'),
        Training_Count=('Training Name', 'nunique')
    ).reset_index()

    # Re-calculate health based on total aggregated hours
    agg_df['Completed_20hrs'] = agg_df['Total_Hours'] >= 20
    
    # Sort or format
    agg_df = agg_df.sort_values(by='Total_Hours', ascending=False)
    
    # Pagination Logic
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
        total_pages=total_pages
    )

@app.route("/analytics")
def analytics():
    df = load_data()
    
    # Custom Date Range filtering logic
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if start_date:
        df = df[pd.to_datetime(df['Start Date'], errors='coerce') >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df['End Date'], errors='coerce') <= pd.to_datetime(end_date)]
    
    # 1. Monthly Trend
    if 'Training Start Month' in df.columns:
        trend_df = df.groupby('Training Start Month').agg(
            total=('Emp ID', 'count'),
            certified=('Completed_20hrs', 'sum')
        ).reset_index()
        months = list(trend_df['Training Start Month'].astype(str))
        trend_total = [int(x) for x in trend_df['total'].values]
        trend_certified = [int(x) for x in trend_df['certified'].values]
    else:
        months, trend_total, trend_certified = [], [], []

    # 2. Dept Cert Rates
    dept_stats = df.groupby('Department').agg(
        total=('Emp ID', 'count'),
        certified=('Completed_20hrs', 'sum')
    ).reset_index()
    dept_stats['cert_rate'] = (dept_stats['certified'] / dept_stats['total'] * 100).fillna(0).round(1)
    
    # Sort by rate descending
    dept_stats = dept_stats.sort_values(by='cert_rate', ascending=False)
    
    dept_labels = list(dept_stats['Department'].astype(str))
    dept_rates = [float(x) for x in dept_stats['cert_rate'].values]

    return render_template(
        "Analytics.html",
        months=months,
        trend_total=trend_total,
        trend_certified=trend_certified,
        dept_labels=dept_labels,
        dept_rates=dept_rates,
        total_records=len(df)
    )

@app.route("/settings")
def settings():
    df = load_data()
    return render_template(
        "settings.html",
        row_count=len(df),
        col_count=len(df.columns),
        columns=list(df.columns)
    )


if __name__ == "__main__":
    app.run(debug=True)