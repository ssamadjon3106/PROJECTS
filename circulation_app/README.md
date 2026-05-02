# Circulation App

Owner: Team Member 3

Purpose: self-service checkout, return, due dates, overdue states, and borrowing history.

Architecture:

- `models.py` stores future circulation policies.
- `forms.py` validates checkout/checkin input.
- `views.py` renders borrowing pages.
- `services.py` contains transactional borrowing logic.
- `selectors.py` contains loan query helpers.
- `urls.py` owns `/team/circulation/`.
- `templates/circulation_app/` owns circulation templates.
- `static/circulation_app/` owns circulation-specific assets.
