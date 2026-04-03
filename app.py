from flask import Flask, render_template, request
import pandas as pd
from urllib.parse import quote_plus

app = Flask(__name__)

# URL encoder for dashboards to handle special chars like ampersand safely
app.jinja_env.filters['urlencode'] = lambda value: quote_plus(str(value))

def load_data():
    df = pd.read_excel("data.xlsx")
    df.columns = [col.strip() for col in df.columns]

    # Trim whitespace and fix values in key columns to avoid URL filter mismatch
    if 'Department' in df.columns:
        df['Department'] = df['Department'].fillna('').astype(str).str.strip()
    if 'Business Unit' in df.columns:
        df['Business Unit'] = df['Business Unit'].fillna('').astype(str).str.strip()

    # Convert hours
    df['Overall Training Duration (Planned Hrs)'] = pd.to_numeric(
        df['Overall Training Duration (Planned Hrs)'], errors='coerce'
    )

    # Completion logic
    df['Completed_20hrs'] = df['Overall Training Duration (Planned Hrs)'] >= 20

    return df


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

    coverage = round((completed_emp / total_emp) * 100, 1) if total_emp > 0 else 0
    completion_rate = coverage

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

    if dept_filter != 'All':
        df = df[df['Department'] == dept_filter]
    if bu_filter != 'All':
        df = df[df['Business Unit'] == bu_filter]

    # Aggregate by employee to remove duplicates
    agg_df = df.groupby(['Emp ID', 'Employee Name', 'Department', 'Business Unit']).agg(
        Total_Hours=('Overall Training Duration (Planned Hrs)', 'sum'),
        Training_Count=('Training Name', 'nunique'),
        Course_List=('Training Name', lambda x: ', '.join(sorted(x.dropna().astype(str).unique())))
    ).reset_index()

    # Re-calculate health based on total aggregated hours
    agg_df['Completed_20hrs'] = agg_df['Total_Hours'] >= 20
    agg_df['Hours_Category'] = agg_df['Total_Hours'].apply(categorize_hours)

    if hours_category and hours_category != 'All':
        agg_df = agg_df[agg_df['Hours_Category'] == hours_category]
    
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
        dept_filter=dept_filter,
        bu_filter=bu_filter,
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
        columns=list(df.columns)
    )


if __name__ == "__main__":
    app.run(debug=True)
