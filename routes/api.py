from flask import Blueprint, request, jsonify
import pandas as pd
from data_processor import load_data, compute_metrics
from genai_helper import (
    generate_query_plan,
    execute_query_plan,
    generate_rag_response,
)

api_bp = Blueprint('api', __name__)

@api_bp.route("/api/insight", methods=["POST"])
def generate_insight():
    body = request.get_json(silent=True) or {}
    context = body.get("context", {}) or {}

    page = str(context.get("page", "dashboard")).strip().lower()
    dept_filter = str(context.get("department", "All")).strip()
    bu_filter = str(context.get("business_unit", "All")).strip()
    start_date = context.get("start_date")
    end_date = context.get("end_date")

    df = load_data()

    if dept_filter and dept_filter != "All" and "Department" in df.columns:
        df = df[df["Department"] == dept_filter]
    if bu_filter and bu_filter != "All" and "Business Unit" in df.columns:
        df = df[df["Business Unit"] == bu_filter]

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

    def _top_series(s: pd.Series, n=3, ascending=False):
        if s is None or len(s) == 0:
            return []
        s2 = s.dropna()
        if len(s2) == 0:
            return []
        s2 = s2.sort_values(ascending=ascending).head(n)
        return [(str(idx), float(val)) for idx, val in s2.items()]

    top_depts = _top_series(metrics.get("dept"), n=3, ascending=False)
    low_depts = _top_series(metrics.get("dept"), n=1, ascending=True)
    top_bus = _top_series(metrics.get("bu"), n=3, ascending=False)
    low_bus = _top_series(metrics.get("bu"), n=1, ascending=True)

    ttype = metrics.get("ttype")
    top_type = None
    if isinstance(ttype, pd.Series) and len(ttype) > 0:
        top_type = (str(ttype.index[0]), float(ttype.iloc[0]))

    emp_agg = df.groupby("Emp ID").agg(
        Total_Hours=("Overall Training Duration (Planned Hrs)", "sum"),
        Employee_Name=("Employee Name", "first") if "Employee Name" in df.columns else ("Emp ID", "first"),
        Department=("Department", "first") if "Department" in df.columns else ("Emp ID", "first"),
        Business_Unit=("Business Unit", "first") if "Business Unit" in df.columns else ("Emp ID", "first"),
    )
    emp_agg["Completed"] = emp_agg["Total_Hours"] >= 20
    remaining = (20 - emp_agg["Total_Hours"]).clip(lower=0)
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

    if page == "employees":
        top_emp = emp_agg.sort_values(by="Total_Hours", ascending=False).head(5)
        lines.append("- **Top employees by total hours**:")
        for eid, row in top_emp.iterrows():
            nm = str(row.get("Employee_Name", "")).strip()
            hrs = float(row.get("Total_Hours", 0))
            lines.append(f"  - {nm} (Emp ID {eid}): {hrs:.1f} hrs")

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

@api_bp.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(silent=True) or {}
    message = body.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided."}), 400

    df = load_data()

    schema_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}

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

    data_result = execute_query_plan(df, gemini_plan)

    final_response = generate_rag_response(message, data_result, gemini_plan)

    return jsonify({
        "query": message,
        "gemini_plan": gemini_plan,
        "data_result": data_result,
        "final_response": final_response,
    })
