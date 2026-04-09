from flask import Flask
from urllib.parse import quote_plus
import os
from project.data_processor import get_active_year

# Import blueprints
from project.routes.dashboard import dashboard_bp
from project.routes.employees import employees_bp
from project.routes.analytics import analytics_bp
from project.routes.api import api_bp
from project.routes.data import data_bp
from project.routes.compare import compare_bp
from project.routes.trainings import trainings_bp

app = Flask(__name__, template_folder="project/templates")
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_training_lnd_42')

# URL encoder for dashboards to handle special chars like ampersand safely
app.jinja_env.filters['urlencode'] = lambda value: quote_plus(str(value))


@app.context_processor
def inject_loaded_year():
    """Expose the active dataset year to all templates."""
    active_year = get_active_year()
    return {
        "loaded_year": active_year,
        "loaded_year_label": active_year if active_year else "Default",
    }

# Register blueprints
app.register_blueprint(dashboard_bp)
app.register_blueprint(employees_bp)
app.register_blueprint(analytics_bp)
app.register_blueprint(api_bp)
app.register_blueprint(data_bp)
app.register_blueprint(compare_bp)
app.register_blueprint(trainings_bp)

if __name__ == "__main__":
    try:
        from project.sync_worker import start_sync_scheduler
        start_sync_scheduler()
    except Exception as e:
        print(f"Could not start sync worker: {e}")

    app.run(debug=True)
