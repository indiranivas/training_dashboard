import os
import re
from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from project.data_processor import set_active_year, get_active_year, set_live_link, get_live_link

data_bp = Blueprint('data', __name__)

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_available_years():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data_files")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    years = set()
    for filename in os.listdir(data_dir):
        match = re.match(r'^data_(\d{4})\.(csv|xlsx|xls)$', filename, re.IGNORECASE)
        if match:
            years.add(match.group(1))
    return sorted(list(years), reverse=True)

@data_bp.route('/data')
def data_page():
    active_year = get_active_year()
    available_years = get_available_years()
    live_link = get_live_link(active_year) if active_year else None
    return render_template('Data.html', active_year=active_year, available_years=available_years, live_link=live_link)

@data_bp.route('/data/set_active', methods=['POST'])
def set_active():
    year = request.form.get('year')
    if year:
        set_active_year(year)
        flash(f'Active year successfully set to {year}.', 'success')
    return redirect(url_for('data.data_page'))

@data_bp.route('/data/set_link', methods=['POST'])
def set_link():
    year = request.form.get('year')
    link = request.form.get('link')
    if year and link:
        set_live_link(year, link)
        set_active_year(year)
        flash(f'Live SharePoint sync link saved for {year}.', 'success')
    else:
        flash('Year and Link are both required.', 'error')
    return redirect(url_for('data.data_page'))

@data_bp.route('/data/upload', methods=['POST'])
def upload_data():
    if 'dataset' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('data.data_page'))
    
    file = request.files['dataset']
    year = request.form.get('year')
    
    if not year or not year.isdigit() or len(year) != 4:
        flash('Invalid year format. Must be a 4-digit number.', 'error')
        return redirect(url_for('data.data_page'))
        
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('data.data_page'))
        
    if file and allowed_file(file.filename):
        ext = file.filename.rsplit('.', 1)[1].lower()
        base_dir = os.path.dirname(os.path.dirname(__file__))
        data_dir = os.path.join(base_dir, "data_files")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
        filename = f"data_{year}.{ext}"
        filepath = os.path.join(data_dir, filename)
        
        try:
            file.save(filepath)
            
            # Automatically set the just-uploaded year as active
            set_active_year(year)
            
            flash(f'Dataset for {year} successfully uploaded and set as active.', 'success')
        except Exception as e:
            flash(f'Error saving file: {str(e)}', 'error')
            
        return redirect(url_for('data.data_page'))
    
    flash('File extension not allowed. Must be CSV or XLSX.', 'error')
    return redirect(url_for('data.data_page'))
