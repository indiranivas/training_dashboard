from flask import Blueprint, render_template, request
import pandas as pd
from data_processor import list_available_years, load_data_for_year, compute_metrics

compare_bp = Blueprint('compare', __name__)

VARIABLE_OPTIONS = {
    "Department": "Department",
    "Business Unit": "Business Unit",
    "Training Type": "Training Type",
    "Training Status": "Training Status",
    "Employee Status": "Employee Status",
    "Designation": "Designation",
    "Training Source": "Training Source",
    "Quarter": "Quarter",
}

METRIC_OPTIONS = {
    "total_hours": "Total Hours",
    "employee_count": "Employees",
    "completed_employees": "Completed Employees (20+ Hrs)",
    "completion_rate": "Completion Rate %",
    "training_count": "Unique Trainings",
    "record_count": "Training Records",
}

QUARTER_ORDER = ["Q1", "Q2", "Q3", "Q4"]


def _safe_unique_union(dfs, column):
    values = set()
    for df in dfs:
        if column in df.columns:
            values.update(
                str(v).strip()
                for v in df[column].dropna().astype(str).tolist()
                if str(v).strip()
            )
    return sorted(values)


def _apply_filters(df, department="All", business_unit="All", training_type="All", training_status="All", quarter="All"):
    if df.empty:
        return df

    filters = [
        ("Department", department),
        ("Business Unit", business_unit),
        ("Training Type", training_type),
        ("Training Status", training_status),
        ("Quarter", quarter),
    ]
    for column, value in filters:
        if value and value != "All" and column in df.columns:
            df = df[df[column] == value]
    return df


def _single_metric_value(df, metric_key):
    if df.empty:
        return 0.0

    hours_col = 'Overall Training Duration (Planned Hrs)'

    if metric_key == "total_hours":
        if hours_col not in df.columns:
            return 0.0
        return round(float(pd.to_numeric(df[hours_col], errors='coerce').fillna(0).sum()), 1)

    if metric_key == "employee_count":
        if 'Emp ID' not in df.columns:
            return 0
        return int(df['Emp ID'].nunique())

    if metric_key == "completed_employees":
        if 'Emp ID' not in df.columns or hours_col not in df.columns:
            return 0
        emp_totals = (
            df.groupby('Emp ID')[hours_col]
            .sum()
            .fillna(0)
        )
        return int((emp_totals >= 20).sum())

    if metric_key == "completion_rate":
        if 'Emp ID' not in df.columns or hours_col not in df.columns:
            return 0.0
        emp_totals = (
            df.groupby('Emp ID')[hours_col]
            .sum()
            .fillna(0)
        )
        total_emp = len(emp_totals)
        if total_emp == 0:
            return 0.0
        return round(float(((emp_totals >= 20).sum() / total_emp) * 100), 1)

    if metric_key == "training_count":
        if 'Training Name' not in df.columns:
            return 0
        return int(df['Training Name'].nunique())

    if metric_key == "record_count":
        return int(len(df))

    return 0.0


def _group_metric_series(df, dimension, metric_key):
    if df.empty or dimension not in df.columns:
        return pd.Series(dtype=float)

    hours_col = 'Overall Training Duration (Planned Hrs)'

    if metric_key == "total_hours" and hours_col in df.columns:
        return (
            df.groupby(dimension)[hours_col]
            .sum()
            .fillna(0)
            .round(1)
        )

    if metric_key == "employee_count" and 'Emp ID' in df.columns:
        return df.groupby(dimension)['Emp ID'].nunique().astype(float)

    if metric_key == "training_count" and 'Training Name' in df.columns:
        return df.groupby(dimension)['Training Name'].nunique().astype(float)

    if metric_key == "record_count":
        return df.groupby(dimension).size().astype(float)

    if metric_key in ("completed_employees", "completion_rate") and 'Emp ID' in df.columns and hours_col in df.columns:
        emp_level = (
            df.groupby([dimension, 'Emp ID'])[hours_col]
            .sum()
            .reset_index()
        )
        emp_level['Completed_20hrs'] = emp_level[hours_col] >= 20
        totals = emp_level.groupby(dimension)['Emp ID'].nunique().astype(float)
        completed = emp_level.groupby(dimension)['Completed_20hrs'].sum().astype(float)
        if metric_key == "completed_employees":
            return completed
        return ((completed / totals.replace(0, pd.NA)) * 100).fillna(0).round(1)

    return pd.Series(dtype=float)


def _build_summary(df):
    metrics = compute_metrics(df)
    return {
        "employees": int(metrics.get("total_emp", 0)),
        "completion_rate": float(metrics.get("completion_rate", 0)),
        "coverage": float(metrics.get("coverage", 0)),
        "total_hours": _single_metric_value(df, "total_hours"),
        "unique_trainings": _single_metric_value(df, "training_count"),
        "completed_employees": _single_metric_value(df, "completed_employees"),
        "records": _single_metric_value(df, "record_count"),
    }


def _format_delta(current, previous, suffix=""):
    delta = round(float(current) - float(previous), 1)
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta}{suffix}"


def _comparison_reference(year, quarter_focus):
    if not year:
        return "—"
    if quarter_focus and quarter_focus != "All":
        return f"{quarter_focus} {year}"
    return str(year)


def _build_comparison_rows(df_a, df_b, dimension, metric_key):
    series_a = _group_metric_series(df_a, dimension, metric_key)
    series_b = _group_metric_series(df_b, dimension, metric_key)

    labels = sorted(set(series_a.index.astype(str)).union(set(series_b.index.astype(str))))
    rows = []
    for label in labels:
        val_a = float(series_a.get(label, 0))
        val_b = float(series_b.get(label, 0))
        delta = round(val_b - val_a, 1)
        delta_pct = 0.0 if val_a == 0 else round((delta / val_a) * 100, 1)
        rows.append({
            "label": label,
            "year_a_value": round(val_a, 1),
            "year_b_value": round(val_b, 1),
            "delta": delta,
            "delta_pct": delta_pct,
            "bigger_value": max(val_a, val_b),
        })

    rows.sort(key=lambda row: (row["bigger_value"], abs(row["delta"])), reverse=True)
    return rows


def _build_quarter_rows(df_a, df_b, metric_key):
    rows = []
    for quarter in QUARTER_ORDER:
        qdf_a = df_a[df_a['Quarter'] == quarter] if 'Quarter' in df_a.columns else pd.DataFrame()
        qdf_b = df_b[df_b['Quarter'] == quarter] if 'Quarter' in df_b.columns else pd.DataFrame()
        val_a = _single_metric_value(qdf_a, metric_key)
        val_b = _single_metric_value(qdf_b, metric_key)
        rows.append({
            "quarter": quarter,
            "year_a_value": round(float(val_a), 1),
            "year_b_value": round(float(val_b), 1),
            "delta": round(float(val_b) - float(val_a), 1),
        })
    return rows


@compare_bp.route("/compare")
def compare():
    available_years = list_available_years()

    # Default: compare the two most distinct (oldest available) years
    # available_years is sorted descending, so [-1] is oldest, [-2] is second oldest
    default_year_a = available_years[-2] if len(available_years) >= 2 else (available_years[0] if available_years else None)
    default_year_b = available_years[-1] if len(available_years) >= 2 else default_year_a
    # If only 2 years, keep smallest as baseline and next as compare
    if default_year_a and default_year_b and default_year_a > default_year_b:
        default_year_a, default_year_b = default_year_b, default_year_a

    year_a = request.args.get('year_a', default_year_a)
    year_b = request.args.get('year_b', default_year_b)
    dimension = request.args.get('dimension', 'Department')
    metric_key = request.args.get('metric', 'completion_rate')
    quarter_focus = request.args.get('quarter', 'All')
    department_filter = request.args.get('department', 'All')
    business_unit_filter = request.args.get('business_unit', 'All')
    training_type_filter = request.args.get('training_type', 'All')
    training_status_filter = request.args.get('training_status', 'All')

    if dimension not in VARIABLE_OPTIONS:
        dimension = 'Department'
    if metric_key not in METRIC_OPTIONS:
        metric_key = 'completion_rate'

    raw_df_a = load_data_for_year(year_a) if year_a else pd.DataFrame()
    raw_df_b = load_data_for_year(year_b) if year_b else pd.DataFrame()

    filter_options = {
        "departments": _safe_unique_union([raw_df_a, raw_df_b], 'Department'),
        "business_units": _safe_unique_union([raw_df_a, raw_df_b], 'Business Unit'),
        "training_types": _safe_unique_union([raw_df_a, raw_df_b], 'Training Type'),
        "training_statuses": _safe_unique_union([raw_df_a, raw_df_b], 'Training Status'),
    }

    df_a = _apply_filters(
        raw_df_a,
        department=department_filter,
        business_unit=business_unit_filter,
        training_type=training_type_filter,
        training_status=training_status_filter,
        quarter=quarter_focus,
    )
    df_b = _apply_filters(
        raw_df_b,
        department=department_filter,
        business_unit=business_unit_filter,
        training_type=training_type_filter,
        training_status=training_status_filter,
        quarter=quarter_focus,
    )

    summary_a = _build_summary(df_a)
    summary_b = _build_summary(df_b)
    comparison_rows = _build_comparison_rows(df_a, df_b, dimension, metric_key)
    quarter_rows = _build_quarter_rows(
        _apply_filters(
            raw_df_a,
            department=department_filter,
            business_unit=business_unit_filter,
            training_type=training_type_filter,
            training_status=training_status_filter,
            quarter="All",
        ),
        _apply_filters(
            raw_df_b,
            department=department_filter,
            business_unit=business_unit_filter,
            training_type=training_type_filter,
            training_status=training_status_filter,
            quarter="All",
        ),
        metric_key,
    )

    top_rows = comparison_rows[:12]
    comparison_ref_a = _comparison_reference(year_a, quarter_focus)
    comparison_ref_b = _comparison_reference(year_b, quarter_focus)
    comparison_scope_label = f"{quarter_focus} comparison view" if quarter_focus != "All" else "Full-year comparison view"

    active_filters = []
    if quarter_focus != "All":
        active_filters.append({"label": "Quarter", "value": quarter_focus})
    if department_filter != "All":
        active_filters.append({"label": "Department", "value": department_filter})
    if business_unit_filter != "All":
        active_filters.append({"label": "Business Unit", "value": business_unit_filter})
    if training_type_filter != "All":
        active_filters.append({"label": "Training Type", "value": training_type_filter})
    if training_status_filter != "All":
        active_filters.append({"label": "Training Status", "value": training_status_filter})

    return render_template(
        "Compare.html",
        available_years=available_years,
        variable_options=VARIABLE_OPTIONS,
        metric_options=METRIC_OPTIONS,
        quarter_options=QUARTER_ORDER,
        year_a=year_a,
        year_b=year_b,
        dimension=dimension,
        metric_key=metric_key,
        quarter_focus=quarter_focus,
        department_filter=department_filter,
        business_unit_filter=business_unit_filter,
        training_type_filter=training_type_filter,
        training_status_filter=training_status_filter,
        filter_options=filter_options,
        summary_a=summary_a,
        summary_b=summary_b,
        summary_deltas={
            "employees": _format_delta(summary_b["employees"], summary_a["employees"]),
            "completion_rate": _format_delta(summary_b["completion_rate"], summary_a["completion_rate"], "%"),
            "coverage": _format_delta(summary_b["coverage"], summary_a["coverage"], "%"),
            "total_hours": _format_delta(summary_b["total_hours"], summary_a["total_hours"]),
            "unique_trainings": _format_delta(summary_b["unique_trainings"], summary_a["unique_trainings"]),
            "completed_employees": _format_delta(summary_b["completed_employees"], summary_a["completed_employees"]),
            # delta booleans for template colouring (year_b is the compare/newer year)
            "employees_up": summary_b["employees"] >= summary_a["employees"],
            "completion_rate_up": summary_b["completion_rate"] >= summary_a["completion_rate"],
            "coverage_up": summary_b["coverage"] >= summary_a["coverage"],
            "total_hours_up": summary_b["total_hours"] >= summary_a["total_hours"],
            "unique_trainings_up": summary_b["unique_trainings"] >= summary_a["unique_trainings"],
            "completed_employees_up": summary_b["completed_employees"] >= summary_a["completed_employees"],
        },
        comparison_rows=comparison_rows,
        quarter_rows=quarter_rows,
        comparison_ref_a=comparison_ref_a,
        comparison_ref_b=comparison_ref_b,
        comparison_scope_label=comparison_scope_label,
        active_filters=active_filters,
        chart_labels=[row["label"] for row in top_rows],
        chart_year_a_values=[row["year_a_value"] for row in top_rows],
        chart_year_b_values=[row["year_b_value"] for row in top_rows],
        quarter_chart_labels=[row["quarter"] for row in quarter_rows],
        quarter_chart_year_a=[row["year_a_value"] for row in quarter_rows],
        quarter_chart_year_b=[row["year_b_value"] for row in quarter_rows],
    )
