from flask import Blueprint, render_template, request
from data_processor import load_data, compute_metrics

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route("/")
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
