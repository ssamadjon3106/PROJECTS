# Seats App

Owner: Team Member 4

Purpose: online seat reservation, seat-map UI, availability checks, and reservation passes.

Architecture:

- `models.py` stores future seat/reservation models if split from `library`.
- `forms.py` validates date, time, duration, and seat filters.
- `views.py` renders seat pages.
- `services.py` contains booking and availability logic.
- `selectors.py` contains seat map query helpers.
- `urls.py` owns `/team/seats/`.
- `templates/seats_app/` owns seat templates.
- `static/seats_app/` owns seat-specific assets.
