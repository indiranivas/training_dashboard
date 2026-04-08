from flask import Blueprint, jsonify, render_template, request
import pandas as pd
from data_processor import load_data
import re

trainings_bp = Blueprint('trainings', __name__)

# ── Column Constants ──────────────────────────────────────────────────────────
COL_SOURCE   = 'Training Source'   # Internal / External / Online
COL_TRAINER  = 'Trainer Name'
COL_TRAINING = 'Training Name'
COL_EMP_NAME = 'Employee Name'
COL_EMP_ID   = 'Emp ID'
COL_HOURS    = 'Overall Training Duration (Planned Hrs)'
COL_STATUS   = 'Training Status'
COL_SOURCE_KEY = '_source_key'
COL_TRAINER_KEY = '_trainer_key'
COL_TRAINING_KEY = '_training_key'
COL_TRAINING_LABEL = '_training_label'


def _normalize_text(value) -> str:
    text = '' if value is None else str(value)
    text = text.replace('\r', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _normalize_training_name(value) -> str:
    text = '' if value is None else str(value)
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\s*/\s*\n\s*', ' / ', text)
    text = re.sub(r'\n+', ' / ', text)
    text = re.sub(r'\s*/\s*', ' / ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _get_df() -> pd.DataFrame:
    """Load and return the active dataset, normalising key columns."""
    df = load_data()
    for col in [COL_SOURCE, COL_TRAINER, COL_TRAINING, COL_EMP_NAME]:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()
    if COL_SOURCE in df.columns:
        df[COL_SOURCE_KEY] = df[COL_SOURCE].map(lambda value: _normalize_text(value).lower())
    if COL_TRAINER in df.columns:
        df[COL_TRAINER_KEY] = df[COL_TRAINER].map(lambda value: _normalize_text(value).lower())
    if COL_TRAINING in df.columns:
        df[COL_TRAINING_LABEL] = df[COL_TRAINING].map(_normalize_training_name)
        df[COL_TRAINING_KEY] = df[COL_TRAINING_LABEL].str.lower()
    return df


# ── Page Route ────────────────────────────────────────────────────────────────
@trainings_bp.route('/trainings')
def trainings_page():
    """Render landing page with summary stats per training source."""
    df = _get_df()

    summary = {}
    if COL_SOURCE in df.columns:
        for src in ['Internal', 'External', 'Online']:
            sub = df[df[COL_SOURCE].str.lower() == src.lower()]
            trainer_count  = sub[COL_TRAINER].nunique()  if COL_TRAINER  in sub.columns else 0
            program_count  = sub[COL_TRAINING].nunique() if COL_TRAINING in sub.columns else 0
            attendee_count = sub[COL_EMP_ID].nunique()   if COL_EMP_ID   in sub.columns else 0
            total_hours    = round(pd.to_numeric(sub[COL_HOURS], errors='coerce').sum(), 1) \
                             if COL_HOURS in sub.columns else 0
            summary[src] = {
                'trainers':  trainer_count,
                'programs':  program_count,
                'attendees': attendee_count,
                'hours':     total_hours,
            }

    return render_template('Trainings.html', current_page='trainings', summary=summary)


# ── API: Get trainers for a source category ───────────────────────────────────
@trainings_bp.route('/trainings/type/<string:category>')
def get_trainers_by_category(category):
    """
    Returns all distinct trainers for a given Training Source,
    along with per-trainer stats: # of programs, # of attendees, total hours.
    """
    df = _get_df()

    if COL_SOURCE not in df.columns:
        return jsonify({"error": "Column 'Training Source' not found in dataset."}), 500

    sub = df[df[COL_SOURCE_KEY] == _normalize_text(category).lower()]
    if sub.empty:
        return jsonify({"error": f"No records found for category '{category}'."}), 404

    if COL_TRAINER not in sub.columns:
        return jsonify({"error": "Column 'Trainer Name' not found in dataset."}), 500

    # Aggregate per trainer
    trainer_agg = (
        sub.groupby(COL_TRAINER)
        .agg(
            programs  = (COL_TRAINING_KEY, 'nunique'),
            attendees = (COL_EMP_ID,   'nunique'),
            hours     = (COL_HOURS,    lambda x: round(pd.to_numeric(x, errors='coerce').sum(), 1))
        )
        .reset_index()
        .sort_values('programs', ascending=False)
    )

    # Remove blank trainer names
    trainer_agg = trainer_agg[trainer_agg[COL_TRAINER] != '']

    trainers = trainer_agg.rename(columns={COL_TRAINER: 'name'}).to_dict(orient='records')
    return jsonify({"category": category, "trainers": trainers})


# ── API: Get training programs for a trainer ──────────────────────────────────
@trainings_bp.route('/trainings/<string:category>/<string:trainer>')
def get_programs_by_trainer(category, trainer):
    """
    Returns all training programs conducted by a specific trainer
    under the given source category, with attendee count and total hours.
    """
    df = _get_df()

    if COL_SOURCE not in df.columns or COL_TRAINER not in df.columns:
        return jsonify({"error": "Required columns not found in dataset."}), 500

    sub = df[
        (df[COL_SOURCE_KEY] == _normalize_text(category).lower()) &
        (df[COL_TRAINER_KEY] == _normalize_text(trainer).lower())
    ]
    if sub.empty:
        return jsonify({"error": f"No records found for trainer '{trainer}' in '{category}'."}), 404

    program_agg = (
        sub.groupby(COL_TRAINING_LABEL)
        .agg(
            attendees = (COL_EMP_ID, 'nunique'),
            hours     = (COL_HOURS,  lambda x: round(pd.to_numeric(x, errors='coerce').sum(), 1))
        )
        .reset_index()
        .sort_values('attendees', ascending=False)
    )

    program_agg = program_agg[program_agg[COL_TRAINING_LABEL] != '']

    # Resolve the canonical trainer name (preserving original casing)
    canonical_trainer = sub[COL_TRAINER].iloc[0]

    programs = program_agg.rename(columns={COL_TRAINING_LABEL: 'name'}).to_dict(orient='records')
    return jsonify({
        "category": category,
        "trainer":  canonical_trainer,
        "programs": programs
    })


# ── API: Get employees who attended a training ────────────────────────────────
@trainings_bp.route('/trainings/attendance')
def get_employees_by_training_query():
    return _get_employees_by_training(
        request.args.get('category', ''),
        request.args.get('trainer', ''),
        request.args.get('training', ''),
    )


@trainings_bp.route('/trainings/<string:category>/<string:trainer>/<path:training>')
def get_employees_by_training(category, trainer, training):
    return _get_employees_by_training(category, trainer, training)


def _get_employees_by_training(category, trainer, training):
    """
    Returns all employees who attended a specific training program
    by a specific trainer under the given source category.
    """
    df = _get_df()

    if not all(c in df.columns for c in [COL_SOURCE, COL_TRAINER, COL_TRAINING]):
        return jsonify({"error": "Required columns not found in dataset."}), 500

    sub = df[
        (df[COL_SOURCE_KEY] == _normalize_text(category).lower()) &
        (df[COL_TRAINER_KEY] == _normalize_text(trainer).lower()) &
        (df[COL_TRAINING_KEY] == _normalize_training_name(training).lower())
    ]
    if sub.empty:
        return jsonify({"error": f"No attendees found for '{training}'."}), 404

    # Build per-employee details — deduplicate by Emp ID, keep first occurrence
    emp_cols = [COL_EMP_ID, COL_EMP_NAME]
    optional = ['Department', 'Business Unit', 'Designation', COL_STATUS]
    for c in optional:
        if c in sub.columns:
            emp_cols.append(c)

    emp_df = (
        sub[emp_cols]
        .drop_duplicates(subset=[COL_EMP_ID])
        .fillna('')
        .sort_values(COL_EMP_NAME)
    )
    # Rename for clean JSON keys
    emp_df = emp_df.rename(columns={
        COL_EMP_ID:   'emp_id',
        COL_EMP_NAME: 'name',
        COL_STATUS:   'status'
    })

    canonical_training = sub[COL_TRAINING_LABEL].iloc[0]
    canonical_trainer  = sub[COL_TRAINER].iloc[0]
    total_hours = round(
        pd.to_numeric(sub[COL_HOURS], errors='coerce').sum(), 1
    ) if COL_HOURS in sub.columns else 0

    return jsonify({
        "category": category,
        "trainer":  canonical_trainer,
        "program":  canonical_training,
        "employees": emp_df.to_dict(orient='records'),
        "count":     len(emp_df),
        "total_hours": total_hours
    })
