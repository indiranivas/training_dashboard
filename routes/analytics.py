from flask import Blueprint, render_template, request
import pandas as pd
from data_processor import load_data, compute_metrics

analytics_bp = Blueprint('analytics', __name__)

@analytics_bp.route("/analytics")
def analytics():
    df = load_data()

    dept_filter = request.args.get('department', 'All')
    bu_filter = request.args.get('business_unit', 'All')
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
