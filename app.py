from flask import Flask
from urllib.parse import quote_plus
import os

# Import blueprints
from routes.dashboard import dashboard_bp
from routes.employees import employees_bp
from routes.analytics import analytics_bp
from routes.api import api_bp
from routes.data import data_bp
from routes.compare import compare_bp
from routes.trainings import trainings_bp

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key_training_lnd_42')

# URL encoder for dashboards to handle special chars like ampersand safely
app.jinja_env.filters['urlencode'] = lambda value: quote_plus(str(value))

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
        from sync_worker import start_sync_scheduler
        start_sync_scheduler()
    except Exception as e:
        print(f"Could not start sync worker: {e}")

    app.run(debug=True)
