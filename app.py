from flask import Flask
from urllib.parse import quote_plus
import os

# Import blueprints
from routes.dashboard import dashboard_bp
from routes.employees import employees_bp
from routes.analytics import analytics_bp
from routes.api import api_bp

app = Flask(__name__)

# URL encoder for dashboards to handle special chars like ampersand safely
app.jinja_env.filters['urlencode'] = lambda value: quote_plus(str(value))

# Register blueprints
app.register_blueprint(dashboard_bp)
app.register_blueprint(employees_bp)
app.register_blueprint(analytics_bp)
app.register_blueprint(api_bp)

if __name__ == "__main__":
    app.run(debug=True)
