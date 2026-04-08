from flask import Blueprint, render_template, request
import pandas as pd
from data_processor import load_data, categorize_hours

employees_bp = Blueprint('employees', __name__)

@employees_bp.route("/employees")
def employees():
    df = load_data()
    departments = sorted([str(d) for d in df['Department'].dropna().unique()])
    business_units = sorted([str(b) for b in df['Business Unit'].dropna().unique()])
    dept_filter = request.args.get('department', 'All')
    bu_filter = request.args.get('business_unit', 'All')
    hours_category = request.args.get('hours_category', 'All')
    status_filter = request.args.get('employee_status', 'All')
    search_query = request.args.get('search_query', '').strip()
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    if start_date:
        df = df[pd.to_datetime(df['Start Date'], errors='coerce') >= pd.to_datetime(start_date)]
    if end_date:
        df = df[pd.to_datetime(df['End Date'], errors='coerce') <= pd.to_datetime(end_date)]

    agg_df = df.groupby('Emp ID').agg(**{
        'Employee Name': ('Employee Name', 'first'),
        'Department': ('Department', 'first'),
        'Business Unit': ('Business Unit', 'first'),
        'Employee_Status': ('Employee Status', 'first'),
        'Total_Hours': ('Overall Training Duration (Planned Hrs)', 'sum'),
        'Training_Count': ('Training Name', 'nunique'),
        'Course_List': ('Training Name', lambda x: ', '.join(sorted(x.dropna().astype(str).unique())))
    }).reset_index()

    agg_df['Completed_20hrs'] = agg_df['Total_Hours'] >= 20
    agg_df['Hours_Category'] = agg_df['Total_Hours'].apply(categorize_hours)

    if dept_filter != 'All':
        agg_df = agg_df[agg_df['Department'] == dept_filter]
    if bu_filter != 'All':
        agg_df = agg_df[agg_df['Business Unit'] == bu_filter]
    if hours_category and hours_category != 'All':
        agg_df = agg_df[agg_df['Hours_Category'] == hours_category]
    if status_filter != 'All':
        agg_df = agg_df[agg_df['Employee_Status'] == status_filter]
        
    if search_query:
        query_lower = search_query.lower()
        id_match = agg_df['Emp ID'].astype(str).str.lower().str.contains(query_lower)
        name_match = agg_df['Employee Name'].astype(str).str.lower().str.contains(query_lower)
        agg_df = agg_df[id_match | name_match]

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

    all_emp_count = load_data()['Emp ID'].nunique()

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
        search_query=search_query,
        page=page,
        total_pages=total_pages
    )
