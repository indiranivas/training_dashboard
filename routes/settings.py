from flask import Blueprint, render_template
from data_processor import load_data

settings_bp = Blueprint('settings', __name__)

@settings_bp.route("/settings")
def settings():
    df = load_data()
    return render_template(
        "settings.html",
        row_count=len(df),
        col_count=len(df.columns),
        columns=list(df.columns),
    )
