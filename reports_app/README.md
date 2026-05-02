# Reports App

Owner: Team Member 5

Purpose: dashboard statistics, admin summaries, borrowing analytics, and export-ready reports.

Architecture:

- `models.py` stores future report snapshots.
- `forms.py` validates report filters.
- `views.py` renders report pages.
- `services.py` contains calculations and exports.
- `selectors.py` contains analytics query helpers.
- `urls.py` owns `/team/reports/`.
- `templates/reports_app/` owns report templates.
- `static/reports_app/` owns report-specific assets.
